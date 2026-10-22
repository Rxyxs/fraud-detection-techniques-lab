"""The same pipeline on real e-commerce fraud: IEEE-CIS (Vesta, Kaggle competition
`ieee-fraud-detection`, 590,540 card-not-present transactions, 3.5% fraud).

The synthetic generator plants its fraud with a clean signature, and the pipeline
scores PR-AUC 0.993 with zero false alarms on it. This script runs every step of
`src.models.train` unchanged (behavioral features, legit-only autoencoder,
SMOTE+Tomek, cost-weighted CatBoost + XGBoost, a CLP-cost threshold tuned on
validation, chronological 70/15/15 split) on real transactions, and then trains
the same two models on IEEE-CIS's own engineered features for comparison.

The mapping to the generator's schema is partial, and that is part of the result:

| Pipeline column | IEEE-CIS source |
|---|---|
| `customer_id` | `card1` + `addr1`, the usual card-holder proxy |
| `timestamp` | `TransactionDT` seconds after 2017-12-01 |
| `amount_clp` | `TransactionAmt` x 650 (USD to CLP, Dec 2017) |
| `merchant_category` | `ProductCD` |
| `latitude`, `longitude` | none: the geo-distance features are constant |

    python -m src.realdata_ieee --zip data/raw/ieee/ieee-fraud-detection.zip
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import zipfile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data.generate_transactions import time_based_split
from src.features.build_features import NUMERIC_FEATURE_COLUMNS, build_feature_matrix
from src.models.autoencoder import train_autoencoder
from src.models.catboost_fraud import (
    compute_sample_weights,
    evaluate,
    find_optimal_threshold,
    resample_train_split,
    train_models,
)
from src.models.train import FULL_FEATURE_COLUMNS, SEED, _add_autoencoder_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "outputs" / "reports" / "real_ieee_metrics.json"
CLP_PER_USD = 650.0
REFERENCE_DATE = pd.Timestamp("2017-12-01")


def load_ieee(zip_path: pathlib.Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z, z.open("train_transaction.csv") as f:
        return pd.read_csv(f)


def to_pipeline_schema(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id": raw["TransactionID"].astype(str),
        "customer_id": raw["card1"].astype(str) + "_" + raw["addr1"].fillna(-1).astype(int).astype(str),
        "timestamp": REFERENCE_DATE + pd.to_timedelta(raw["TransactionDT"], unit="s"),
        "amount_clp": raw["TransactionAmt"] * CLP_PER_USD,
        "merchant_category": raw["ProductCD"].astype(str),
        "latitude": 0.0,
        "longitude": 0.0,
        "is_fraud": raw["isFraud"].astype(int),
    })


def native_features(raw: pd.DataFrame) -> pd.DataFrame:
    """IEEE-CIS's own numeric columns; categorical ones as integer codes (NaN -> -1)."""
    X = raw.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category").cat.codes
    return X.astype("float32")


def run_pipeline(features: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Every step of src.models.train.main, on already-built features."""
    train_df, val_df, test_df = time_based_split(features)
    scaler = StandardScaler().fit(train_df[NUMERIC_FEATURE_COLUMNS].to_numpy())
    legit = (train_df["is_fraud"] == 0).to_numpy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae = train_autoencoder(scaler.transform(train_df[NUMERIC_FEATURE_COLUMNS].to_numpy())[legit],
                           n_features=len(NUMERIC_FEATURE_COLUMNS), device=device, seed=SEED)
    train_df, val_df, test_df = _add_autoencoder_score(train_df, val_df, test_df, scaler, ae.model, device)

    X_res, y_res = resample_train_split(train_df[FULL_FEATURE_COLUMNS], train_df["is_fraud"].to_numpy(), seed=SEED)
    models = train_models(X_res, y_res, sample_weight_train=compute_sample_weights(y_res, X_res["amount_clp"].to_numpy()),
                          seed=SEED)

    def ensemble(df):
        return (models.catboost.predict_proba(df[FULL_FEATURE_COLUMNS])[:, 1]
                + models.xgboost.predict_proba(df[FULL_FEATURE_COLUMNS])[:, 1]) / 2.0

    threshold, _ = find_optimal_threshold(val_df["is_fraud"].to_numpy(), ensemble(val_df), val_df["amount_clp"].to_numpy())
    proba = ensemble(test_df)
    metrics = evaluate(test_df["is_fraud"].to_numpy(), proba, test_df["amount_clp"].to_numpy(), threshold)
    metrics["threshold"] = threshold
    return metrics, test_df.assign(proba=proba)


def run_native(raw: pd.DataFrame) -> dict:
    """Same CatBoost + XGBoost (no SMOTE: the native columns are full of NaN), same split,
    same cost threshold, on IEEE-CIS's own features."""
    df = raw.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df)
    tr, va = int(n * 0.70), int(n * 0.85)
    X, y = native_features(df), df["isFraud"].to_numpy()
    amount = (df["TransactionAmt"] * CLP_PER_USD).to_numpy()
    models = train_models(X.iloc[:tr], y[:tr], sample_weight_train=compute_sample_weights(y[:tr], amount[:tr]), seed=SEED)

    def ensemble(sl):
        return (models.catboost.predict_proba(X.iloc[sl])[:, 1] + models.xgboost.predict_proba(X.iloc[sl])[:, 1]) / 2.0

    threshold, _ = find_optimal_threshold(y[tr:va], ensemble(slice(tr, va)), amount[tr:va])
    metrics = evaluate(y[va:], ensemble(slice(va, n)), amount[va:], threshold)
    metrics["threshold"] = threshold
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", type=pathlib.Path, default=ROOT / "data" / "raw" / "ieee" / "ieee-fraud-detection.zip")
    args = parser.parse_args()

    start = time.perf_counter()
    raw = load_ieee(args.zip)
    print(f"IEEE-CIS: {len(raw):,} transactions, fraud {raw['isFraud'].mean():.3%}")
    features = build_feature_matrix(to_pipeline_schema(raw))
    print("Pipeline on the behavioral features...")
    pipeline_metrics, test_df = run_pipeline(features)
    print(json.dumps(pipeline_metrics, indent=2, default=float))
    print("Same models on IEEE-CIS's native features...")
    native_metrics = run_native(raw)
    print(json.dumps(native_metrics, indent=2, default=float))

    y = test_df["is_fraud"].to_numpy()
    summary = {
        "dataset": "ieee-fraud-detection (train_transaction.csv)",
        "n_transactions": int(len(raw)),
        "fraud_rate": round(float(raw["isFraud"].mean()), 4),
        "test_rows": int(len(test_df)),
        "test_frauds": int(y.sum()),
        "pipeline_behavioral_features": pipeline_metrics,
        "pipeline_pr_auc_check": round(float(average_precision_score(y, test_df["proba"])), 4),
        "pipeline_roc_auc_check": round(float(roc_auc_score(y, test_df["proba"])), 4),
        "same_models_native_features": native_metrics,
        "elapsed_seconds": round(time.perf_counter() - start, 1),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
