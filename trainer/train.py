# -*- coding: utf-8 -*-
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import psycopg2
from xgboost import XGBRanker


def get_conn():
    return psycopg2.connect(
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
    )


def build_feature_cols() -> List[str]:
    cols = []
    for r in range(1, 14):
        cols.extend([
            f"rank{r}_oddset_right",
            f"rank{r}_oddset_even",
            f"rank{r}_oddset_wrong",
        ])
    return cols


def load_training_data() -> Tuple[pd.DataFrame, List[int]]:
    feat_cols = build_feature_cols()
    select_cols = ["omgang_id", "correct"] + feat_cols
    sql = f"SELECT {', '.join(select_cols)} FROM historik WHERE correct IS NOT NULL ORDER BY omgang_id"
    with get_conn() as conn:
        df = pd.read_sql_query(sql, conn)
    # Ensure ints/floats and fill NaNs with 0 for flags
    df[feat_cols] = df[feat_cols].fillna(0).astype(np.float32)
    df["correct"] = df["correct"].astype(np.int32)

    # Build groups per omgang_id and filter groups with >1 rows (ranker needs comparisons)
    counts = df.groupby("omgang_id").size()
    valid_ids = counts[counts > 1].index
    df = df[df["omgang_id"].isin(valid_ids)].copy()
    groups = df.groupby("omgang_id").size().astype(int).tolist()
    return df, groups


def train_and_save(df: pd.DataFrame, groups: List[int]) -> None:
    feat_cols = build_feature_cols()
    X = df[feat_cols].values
    y = df["correct"].values

    model = XGBRanker(
        objective="rank:ndcg",
        n_estimators=250,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        n_jobs=0,
    )

    # Fit with group sizes
    model.fit(X, y, group=groups)

    os.makedirs("/models", exist_ok=True)
    model_path = "/models/xgb_ranker.json"
    model.save_model(model_path)

    # Save feature names for reference
    with open("/models/feature_columns.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(feat_cols))

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    try:
        df, groups = load_training_data()
        if df.empty or not groups:
            print("No training data found in historik with required features.")
            sys.exit(2)
        print(f"Training on {len(df)} rows across {len(groups)} groups...")
        train_and_save(df, groups)
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)
