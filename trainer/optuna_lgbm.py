# -*- coding: utf-8 -*-
import os
import sys
import argparse
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import psycopg2
import optuna
import lightgbm as lgb
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
        "unik_flagga",
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
    for r in range(1, 14):
        cols.append(f"rank{r}_oddset_tio_tidningar")
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
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    df["correct"] = df["correct"].astype(np.int32)

    # group sizes per omgang
    counts = df.groupby("omgang_id").size()
    valid_ids = counts[counts > 1].index  # require >1 rows per group for ranking
    df = df[df["omgang_id"].isin(valid_ids)].copy()
    groups = df.groupby("omgang_id").size().astype(int).tolist()
    return df, groups


def make_features_sparse(df: pd.DataFrame) -> csr_matrix:
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
    return X_sparse


def split_by_omgang(df: pd.DataFrame, valid_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    ids = df["omgang_id"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_valid = max(1, int(round(len(ids) * valid_frac)))
    valid_ids = set(ids[:n_valid])
    train_df = df[~df["omgang_id"].isin(valid_ids)].copy()
    valid_df = df[df["omgang_id"].isin(valid_ids)].copy()
    train_groups = train_df.groupby("omgang_id").size().astype(int).tolist()
    valid_groups = valid_df.groupby("omgang_id").size().astype(int).tolist()
    return train_df, valid_df, train_groups, valid_groups


def objective_factory(df: pd.DataFrame, eval_at: List[int], early_stopping: int, seed: int):
    def objective(trial: optuna.Trial) -> float:
        train_df, valid_df, train_groups, valid_groups = split_by_omgang(df, valid_frac=0.2, seed=seed + trial.number)

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 5),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "random_state": seed,
            "n_jobs": -1,
            "eval_at": eval_at,
        }

        X_train = make_features_sparse(train_df)
        y_train = train_df["correct"].values
        X_valid = make_features_sparse(valid_df)
        y_valid = valid_df["correct"].values

        model = LGBMRanker(**params)
        callbacks = [
            lgb.early_stopping(early_stopping, verbose=False),
        ]
        model.fit(
            X_train,
            y_train,
            group=train_groups,
            eval_set=[(X_valid, y_valid)],
            eval_group=[valid_groups],
            callbacks=callbacks,
        )

        # Prefer last eval_at entry if present (e.g., ndcg@13); else max available ndcg
        best_scores = model.booster_.best_score.get("valid_0", {})
        score = None
        # e.g., keys like 'ndcg@5', 'ndcg@10', 'ndcg@13'
        for k in [f"ndcg@{eval_at[-1]}" if eval_at else "ndcg"]:
            if k in best_scores:
                score = best_scores[k]
                break
        if score is None:
            # fallback: take max of any ndcg key
            ndcgs = [v for mk, v in best_scores.items() if mk.startswith("ndcg")]
            score = max(ndcgs) if ndcgs else 0.0
        return float(score)

    return objective


def main():
    ap = argparse.ArgumentParser(description="Optuna hyperparameteroptimering för LGBMRanker (ranking)")
    ap.add_argument("--trials", type=int, default=50, help="Antal Optuna-trials (default 50)")
    ap.add_argument("--omgang_id_min", type=int, default=None, help="Min omgang_id att inkludera")
    ap.add_argument("--omgang_id_max", type=int, default=None, help="Max omgang_id att inkludera")
    ap.add_argument("--valid-frac", type=float, default=0.2, help="Andel grupper (omgångar) i validering (default 0.2)")
    ap.add_argument("--early-stopping", type=int, default=50, help="Early stopping rounds (default 50)")
    ap.add_argument("--eval-at", type=int, nargs="+", default=[13], help="eval_at nivåer för NDCG, t.ex. 13 eller 5 10 13")
    ap.add_argument("--seed", type=int, default=42, help="Slumptalsfrö")
    ap.add_argument("--save-model", action="store_true", help="Träna om på all data med bästa parametrar och spara modell")
    args = ap.parse_args()

    try:
        df, _ = load_training_data(args.omgang_id_min, args.omgang_id_max)
        if df.empty:
            print("No training data found in historik with required features.")
            sys.exit(2)

        # Bind objective with provided settings
        def obj(trial: optuna.Trial) -> float:
            # re-split per trial to reduce variance; reuse df filtered by args
            return objective_factory(df, args.eval_at, args.early_stopping, args.seed)(trial)

        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=args.trials)

        print("Best value (NDCG):", study.best_value)
        print("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        # Optionally refit on all data with best params and save
        if args.save_model:
            best_params = {
                **study.best_params,
                "objective": "lambdarank",
                "metric": "ndcg",
                "random_state": args.seed,
                "n_jobs": -1,
                "eval_at": args.eval_at,
            }
            # Build full dataset
            groups = df.groupby("omgang_id").size().astype(int).tolist()
            X_all = make_features_sparse(df)
            y_all = df["correct"].values
            model = LGBMRanker(**best_params)
            model.fit(X_all, y_all, group=groups)

            os.makedirs("/models", exist_ok=True)
            model_path = "/models/lgbm_ranker.txt"
            model.booster_.save_model(model_path)

            # Save feature columns for predicter compatibility
            feat_cols = build_feature_cols()
            with open("/models/feature_columns.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(feat_cols))

            # Save best params snapshot
            with open("/models/lgbm_ranker_optuna.json", "w", encoding="utf-8") as f:
                json.dump({
                    "best_value": study.best_value,
                    "best_params": study.best_params,
                    "eval_at": args.eval_at,
                }, f, indent=2)

            print(f"Saved tuned model to {model_path} and params to /models/lgbm_ranker_optuna.json")

    except Exception as e:
        print(f"Optuna tuning failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
