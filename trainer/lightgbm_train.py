# -*- coding: utf-8 -*-
import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import psycopg2
from lightgbm import LGBMRanker
from scipy.sparse import csr_matrix, hstack


def get_conn():
    return psycopg2.connect(
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
    )


def build_feature_cols() -> List[str]:
    cols = [
        "oddset_radsumma",
        "oddset_radsumma_max",
        "oddset_radsumma_min",
        "oddset_right_count",
        "oddset_even_count",
        "oddset_wrong_count",
        "oddset_right_points",
        "oddset_even_points",
        "oddset_wrong_points",
        "oddset_group1_right_count",
        "oddset_group1_even_count",
        "oddset_group1_wrong_count",
        "oddset_group2_right_count",
        "oddset_group2_even_count",
        "oddset_group2_wrong_count",
        "oddset_group3_right_count",
        "oddset_group3_even_count",
        "oddset_group3_wrong_count",
    ]
    for r in range(1, 14):
        cols.extend([
            f"rank{r}_oddset_right",
            f"rank{r}_oddset_even",
            f"rank{r}_oddset_wrong",
        ])
    return cols


def load_training_data(omgang_id_min: int | None = None, omgang_id_max: int | None = None) -> Tuple[pd.DataFrame, List[int]]:
    feat_cols = build_feature_cols()
    select_cols = ["omgang_id", "correct"] + feat_cols
    base_sql = f"SELECT {', '.join(select_cols)} FROM historik WHERE correct IS NOT NULL"
    where_parts = []
    params: list[object] = []
    if omgang_id_min is not None:
        where_parts.append("omgang_id >= %s")
        params.append(omgang_id_min)
    if omgang_id_max is not None:
        where_parts.append("omgang_id <= %s")
        params.append(omgang_id_max)
    if where_parts:
        base_sql += " AND " + " AND ".join(where_parts)
    sql = base_sql + " ORDER BY omgang_id"
    with get_conn() as conn:
        df = pd.read_sql_query(sql, conn, params=tuple(params) if params else None)
    df[feat_cols] = df[feat_cols].fillna(0).astype(np.float32)
    df["correct"] = df["correct"].astype(np.int32)

    counts = df.groupby("omgang_id").size()
    valid_ids = counts[counts > 1].index
    df = df[df["omgang_id"].isin(valid_ids)].copy()
    groups = df.groupby("omgang_id").size().astype(int).tolist()
    return df, groups


def train_and_save(df: pd.DataFrame, groups: List[int]) -> None:
    feat_cols = build_feature_cols()
    rank_flag_cols = [
        f"rank{r}_oddset_{k}"
        for r in range(1, 14)
        for k in ("right", "even", "wrong")
    ]
    dense_cols = [c for c in feat_cols if c not in rank_flag_cols]

    X_dense = df[dense_cols].astype(np.float32).values
    X_flags = df[rank_flag_cols].astype(np.float32).values

    X_sparse = hstack([csr_matrix(X_dense), csr_matrix(X_flags)], format="csr")
    y = df["correct"].values

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        feature_fraction=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_sparse, y, group=groups)

    os.makedirs("/models", exist_ok=True)
    model_path = "/models/lgbm_ranker.txt"
    # Save LightGBM native model
    model.booster_.save_model(model_path)

    with open("/models/feature_columns.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(feat_cols))

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train LGBMRanker on historik with optional omgang_id filtering")
    ap.add_argument("--omgang_id_min", type=int, default=None, help="Minimum omgang_id to include")
    ap.add_argument("--omgang_id_max", type=int, default=None, help="Maximum omgang_id to include")
    args = ap.parse_args()

    try:
        df, groups = load_training_data(args.omgang_id_min, args.omgang_id_max)
        if df.empty or not groups:
            print("No training data found in historik with required features.")
            sys.exit(2)
        rng_txt = []
        if args.omgang_id_min is not None:
            rng_txt.append(f"min={args.omgang_id_min}")
        if args.omgang_id_max is not None:
            rng_txt.append(f"max={args.omgang_id_max}")
        rng_str = f" (omgang_id {' & '.join(rng_txt)})" if rng_txt else ""
        print(f"Training on {len(df)} rows across {len(groups)} groups{rng_str}...")
        train_and_save(df, groups)
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)
