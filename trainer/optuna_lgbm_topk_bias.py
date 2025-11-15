# -*- coding: utf-8 -*-
import os
import sys
import argparse
import json
from typing import List, Tuple, Optional, Dict

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


def load_training_data(omgang_id_min: Optional[int] = None, omgang_id_max: Optional[int] = None) -> Tuple[pd.DataFrame, List[int]]:
    feat_cols = build_feature_cols()
    select_cols = ["omgang_id", "correct"] + feat_cols
    base_sql = f"SELECT {', '.join(select_cols)} FROM historik WHERE correct IS NOT NULL"
    where_parts: List[str] = []
    params: List[object] = []
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

    counts = df.groupby("omgang_id").size()
    valid_ids = counts[counts > 1].index
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


def apply_orc_transform(df: pd.DataFrame, scheme: str) -> None:
    col = "oddset_right_count"
    if col not in df.columns:
        return
    x = df[col].astype(np.float32).to_numpy(copy=False)
    if scheme == "none":
        return
    elif scheme == "cap10":
        x = np.minimum(x, 10.0)
    elif scheme == "sqrt":
        x = np.sqrt(np.maximum(x, 0.0))
    elif scheme == "log1p":
        x = np.log1p(np.maximum(x, 0.0))
    else:
        raise ValueError(f"Unknown orc-transform scheme: {scheme}")
    df[col] = x.astype(np.float32)


def compute_sample_weights(df: pd.DataFrame, scheme: str) -> np.ndarray:
    if scheme == "none":
        return np.ones(len(df), dtype=np.float32)
    col = "oddset_right_count"
    if col not in df.columns:
        return np.ones(len(df), dtype=np.float32)

    count = df[col].astype(np.float32).to_numpy(copy=False)
    labels = df["correct"].astype(np.int32).to_numpy(copy=False)

    # Base weights: penalize very high counts softly above threshold
    thr = 10.0
    min_w = 0.6 if scheme == "mild" else 0.5
    w = np.ones_like(count, dtype=np.float32)
    over = np.clip(count - thr, 0.0, None)
    # Decrease linearly up to count ~13
    span = 3.0
    decay = (over / span)
    w = w * (1.0 - decay)
    w = np.clip(w, min_w, 1.0)

    # Boost moderate counts that still have high labels (>=12)
    moderate = (count >= 6) & (count <= 9) & (labels >= 12)
    boost = 0.2 if scheme == "mild" else 0.3
    w[moderate] = np.minimum(w[moderate] + boost, 1.5)

    # Normalize per group to mean 1.0 to keep group balance
    gids = df["omgang_id"].to_numpy()
    # Compute mean per group
    _, inv, cnts = np.unique(gids, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=w)
    means = sums / cnts
    w = w / means[inv]
    return w.astype(np.float32)


def build_label_gain(scheme: str, max_label: int) -> Optional[List[float]]:
    if scheme == "default":
        return None
    gains = [0.0] * (max_label + 1)
    if scheme == "balanced":
        mapping = {10: 1.0, 11: 2.0, 12: 3.0, 13: 6.0}
    elif scheme == "conservative":
        mapping = {10: 1.0, 11: 1.5, 12: 2.5, 13: 5.0}
    else:
        raise ValueError(f"Unknown label_gain scheme: {scheme}")
    for k, v in mapping.items():
        if k <= max_label:
            gains[k] = float(v)
    return gains


def compute_recall_at_k(valid_df: pd.DataFrame, preds: np.ndarray, k: int) -> Dict[str, float]:
    gids = valid_df["omgang_id"].values
    labels = valid_df["correct"].values
    order = np.argsort(gids, kind="stable")
    gids_sorted = gids[order]
    labels_sorted = labels[order]
    preds_sorted = preds[order]
    unique_ids, start_idx = np.unique(gids_sorted, return_index=True)
    end_idx = np.append(start_idx[1:], len(gids_sorted))
    groups_with_13 = 0
    groups_13_captured = 0
    recall12_sum = 0.0
    recall12_groups = 0
    for s, e in zip(start_idx, end_idx):
        lbls = labels_sorted[s:e]
        pr = preds_sorted[s:e]
        topn = min(k, len(lbls))
        top_idx = np.argpartition(-pr, range(topn))[:topn]
        denom13 = np.sum(lbls == 13)
        if denom13 > 0:
            groups_with_13 += 1
            if np.any(lbls[top_idx] == 13):
                groups_13_captured += 1
        denom12 = np.sum(lbls >= 12)
        if denom12 > 0:
            numer12 = int(np.sum(lbls[top_idx] >= 12))
            recall12_sum += (numer12 / float(denom12))
            recall12_groups += 1
    rec_13 = (groups_13_captured / groups_with_13) if groups_with_13 > 0 else float("nan")
    rec_ge12 = (recall12_sum / recall12_groups) if recall12_groups > 0 else float("nan")
    return {"recall13@k": rec_13, "recall_ge12@k": rec_ge12}


def objective_factory(df: pd.DataFrame, eval_at: List[int], early_stopping: int, seed: int,
                      truncation_choices: List[int], label_gain_scheme: str, topk: int,
                      orc_transform: str, weight_scheme: str):
    max_label = int(df["correct"].max()) if not df.empty else 13
    label_gain = build_label_gain(label_gain_scheme, max_label)

    def objective(trial: optuna.Trial) -> float:
        train_df, valid_df, train_groups, valid_groups = split_by_omgang(df, valid_frac=0.2, seed=seed + trial.number)

        # Apply oddset_right_count transform
        apply_orc_transform(train_df, orc_transform)
        apply_orc_transform(valid_df, orc_transform)

        # Sample weights for training only
        train_w = compute_sample_weights(train_df, weight_scheme)

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "num_leaves": trial.suggest_int("num_leaves", 63, 255),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 5),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.9),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 0.5, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "lambdarank_truncation_level": trial.suggest_categorical("lambdarank_truncation_level", truncation_choices),
            "random_state": seed,
            "n_jobs": -1,
            "eval_at": eval_at,
        }
        if label_gain is not None:
            params["label_gain"] = label_gain

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
            sample_weight=train_w,
            callbacks=callbacks,
        )

        # Trial score: NDCG at the last eval_at
        best_scores = model.booster_.best_score.get("valid_0", {})
        score = None
        for k in [f"ndcg@{eval_at[-1]}" if eval_at else "ndcg"]:
            if k in best_scores:
                score = best_scores[k]
                break
        if score is None:
            ndcgs = [v for mk, v in best_scores.items() if mk.startswith("ndcg")]
            score = max(ndcgs) if ndcgs else 0.0

        # Auxiliary logging: recall@topk for label==13 and label>=12
        preds = model.predict(X_valid, num_iteration=model.best_iteration_)
        metrics = compute_recall_at_k(valid_df, preds, k=topk)
        print(
            f"Trial {trial.number}: ndcg@{eval_at[-1]}={score:.6f} | recall13@{topk}={metrics['recall13@k']:.4f} | recall>=12@{topk}={metrics['recall_ge12@k']:.4f}"
        )

        return float(score)

    return objective


def main():
    ap = argparse.ArgumentParser(description="Optuna LGBMRanker (top-K) med bias-kontroll för oddset_right_count")
    ap.add_argument("--trials", type=int, default=120, help="Antal Optuna-trials (default 120)")
    ap.add_argument("--omgang_id_min", type=int, default=None, help="Min omgang_id att inkludera")
    ap.add_argument("--omgang_id_max", type=int, default=None, help="Max omgang_id att inkludera")
    ap.add_argument("--valid-frac", type=float, default=0.2, help="Andel grupper (omgångar) i validering (default 0.2)")
    ap.add_argument("--early-stopping", type=int, default=75, help="Early stopping rounds (default 75)")
    ap.add_argument("--eval-at", type=int, nargs="+", default=[200, 500], help="eval_at nivåer för NDCG, sista används för Optuna-score")
    ap.add_argument("--seed", type=int, default=42, help="Slumptalsfrö")
    ap.add_argument("--save-model", action="store_true", help="Träna om på all data med bästa parametrar och spara modell")
    ap.add_argument("--label-gain-scheme", choices=["default", "balanced", "conservative"], default="balanced", help="Gain-schema för labels (default balanced)")
    ap.add_argument("--truncation-levels", type=int, nargs="+", default=[300, 400, 500, 600], help="Kandidater för lambdarank_truncation_level")
    ap.add_argument("--topk", type=int, default=500, help="K-värde för recall-rapportering (default 500)")
    ap.add_argument("--orc-transform", choices=["none", "cap10", "sqrt", "log1p"], default="cap10", help="Transform av oddset_right_count (default cap10)")
    ap.add_argument("--weight-scheme", choices=["none", "mild", "medium"], default="mild", help="Viktningsschema för sample_weight (default mild)")
    args = ap.parse_args()

    try:
        df, _ = load_training_data(args.omgang_id_min, args.omgang_id_max)
        if df.empty:
            print("No training data found in historik with required features.")
            sys.exit(2)

        def obj(trial: optuna.Trial) -> float:
            return objective_factory(
                df,
                args.eval_at,
                args.early_stopping,
                args.seed,
                args.truncation_levels,
                args.label_gain_scheme,
                args.topk,
                args.orc_transform,
                args.weight_scheme,
            )(trial)

        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=args.trials)

        print("Best value (NDCG):", study.best_value)
        print("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        if args.save_model:
            max_label = int(df["correct"].max()) if not df.empty else 13
            label_gain = build_label_gain(args.label_gain_scheme, max_label)
            best_params = {
                **study.best_params,
                "objective": "lambdarank",
                "metric": "ndcg",
                "random_state": args.seed,
                "n_jobs": -1,
                "eval_at": args.eval_at,
            }
            if label_gain is not None:
                best_params["label_gain"] = label_gain

            # Apply transform and weights on full data
            apply_orc_transform(df, args.orc_transform)
            groups = df.groupby("omgang_id").size().astype(int).tolist()
            X_all = make_features_sparse(df)
            y_all = df["correct"].values
            w_all = compute_sample_weights(df, args.weight_scheme)

            model = LGBMRanker(**best_params)
            model.fit(X_all, y_all, group=groups, sample_weight=w_all)

            os.makedirs("/models", exist_ok=True)
            model_path = "/models/lgbm_ranker.txt"
            model.booster_.save_model(model_path)

            feat_cols = build_feature_cols()
            with open("/models/feature_columns.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(feat_cols))

            snapshot = {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "eval_at": args.eval_at,
                "label_gain_scheme": args.label_gain_scheme,
                "orc_transform": args.orc_transform,
                "weight_scheme": args.weight_scheme,
            }
            if label_gain is not None:
                snapshot["label_gain"] = label_gain
            with open("/models/lgbm_ranker_optuna_topk_bias.json", "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)

            print("Saved tuned model to /models/lgbm_ranker.txt and params to /models/lgbm_ranker_optuna_topk_bias.json")

    except Exception as e:
        print(f"Optuna tuning failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

