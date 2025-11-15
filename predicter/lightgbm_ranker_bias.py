import os
import sys
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import lightgbm as lgb


def get_conn():
    return psycopg2.connect(
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
    )


def load_feature_cols_from_models(models_dir: str = "/models") -> List[str]:
    path = os.path.join(models_dir, "feature_columns.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hittar inte feature_columns.txt i {models_dir}")
    with open(path, "r", encoding="utf-8") as f:
        cols = [line.strip() for line in f if line.strip()]
    return cols


def ensure_prediction_column(cur) -> None:
    cur.execute(
        """
        ALTER TABLE kommande
        ADD COLUMN IF NOT EXISTS rank_predict_1 REAL
        """
    )


def fetch_kommande_batch(conn, feat_cols: List[str], omgang_id: Optional[int], last_id: int, limit: int):
    select_cols = ["id", "omgang_id", "rad"] + feat_cols
    cols_sql = ", ".join(select_cols)
    where = ["(rank_predict_1 IS NULL)"]
    params: List[object] = []
    if omgang_id is not None:
        where.append("omgang_id = %s")
        params.append(omgang_id)
    if last_id:
        where.append("id > %s")
        params.append(last_id)
    where_sql = " AND ".join(where)
    sql = f"SELECT {cols_sql} FROM kommande WHERE {where_sql} ORDER BY id LIMIT %s"
    params.append(limit)
    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
        colnames = [d[0] for d in cur.description]
    return rows, colnames


def update_predictions(conn, updates: List[tuple]):
    if not updates:
        return
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            UPDATE kommande AS c
            SET rank_predict_1 = v.score
            FROM (VALUES %s) AS v(id, score)
            WHERE c.id = v.id
            """,
            updates,
        )
        conn.commit()


def apply_orc_transform_inplace(df: pd.DataFrame, scheme: str) -> None:
    col = "oddset_right_count"
    if col not in df.columns or scheme == "none":
        return
    x = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    if scheme == "cap10":
        x = np.minimum(x, 10.0)
    elif scheme == "sqrt":
        x = np.sqrt(np.maximum(x, 0.0))
    elif scheme == "log1p":
        x = np.log1p(np.maximum(x, 0.0))
    else:
        raise ValueError(f"Unknown orc-transform scheme: {scheme}")
    df[col] = x.astype(np.float32)


def predict_for_kommande(omgang_id: Optional[int], batch_size: int, orc_transform: str) -> None:
    feat_cols = load_feature_cols_from_models()
    model_path = "/models/lgbm_ranker.txt"
    booster = lgb.Booster(model_file=model_path)

    with get_conn() as conn:
        with conn.cursor() as cur:
            ensure_prediction_column(cur)
        conn.commit()

        processed = 0
        last_id = 0
        while True:
            rows, colnames = fetch_kommande_batch(conn, feat_cols, omgang_id, last_id, batch_size)
            if not rows:
                break
            df = pd.DataFrame(rows, columns=colnames)
            # Apply the same transform as training (row-wise, batch-friendly)
            apply_orc_transform_inplace(df, orc_transform)
            X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
            preds = booster.predict(X)
            updates = [(int(row_id), float(score)) for row_id, score in zip(df["id"].tolist(), preds.tolist())]
            update_predictions(conn, updates)
            processed += len(updates)
            last_id = int(df["id"].iloc[-1])
            print(f"Predicter (LightGBM bias): uppdaterade {processed} rader... (senaste id={last_id})")

    print("Predicter (LightGBM bias): klart.")


def main():
    ap = argparse.ArgumentParser(description="Predicera rank-score (LGBM) med oddset_right_count-transform")
    ap.add_argument("--omgang-id", type=int, default=None, help="Om satt: begränsa till denna omgång")
    ap.add_argument("--batch-size", type=int, default=5000, help="Antal rader per batch vid uppdatering")
    ap.add_argument("--orc-transform", choices=["none", "cap10", "sqrt", "log1p"], default="cap10", help="Transform av oddset_right_count som matchar träningen")
    args = ap.parse_args()

    try:
        predict_for_kommande(args.omgang_id, args.batch_size, args.orc_transform)
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

