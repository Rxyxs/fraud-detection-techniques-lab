"""Combines time-velocity, geo-distance, and amount-deviation feature
engineering into the final numeric feature matrix used by the models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.geo_distance import add_geo_features
from src.features.time_velocity import add_time_velocity_features

# Columns fed to the autoencoder / CatBoost / XGBoost models. Kept as an
# explicit, ordered list so training and online inference (the API) can never
# silently drift apart.
NUMERIC_FEATURE_COLUMNS = [
    "amount_clp",
    "amount_zscore",
    "hour_of_day",
    "day_of_week",
    "seconds_since_prev",
    "velocity_ratio",
    "txn_count_last_1h",
    "txn_count_last_24h",
    "amount_sum_last_1h",
    "distance_from_prev_km",
    "implied_speed_kmh",
    "is_impossible_travel",
    "distance_from_home_km",
]

# seconds_since_prev is +inf for a customer's very first transaction; a finite
# but large sentinel keeps every downstream model (scaler, autoencoder,
# CatBoost/XGBoost) well-defined without special-casing infinities.
MAX_SECONDS_SINCE_PREV = 30 * 24 * 3600.0  # 30 days

# amount_zscore and velocity_ratio are ratios with a customer's own historical
# std/mean in the denominator; a customer with only 1-2 prior transactions can
# produce a near-zero denominator and blow the ratio up to an arbitrarily
# large, non-informative value (observed empirically: uncapped z-scores up to
# ~1.7e5). Clipping (winsorizing) keeps these features numerically well-posed
# for the StandardScaler/autoencoder without discarding the real fraud signal,
# which lives at moderate-to-large values (single/low double digits), not at
# these small-sample-noise extremes.
AMOUNT_ZSCORE_CAP = 30.0
VELOCITY_RATIO_CAP = 50.0
# Cap implied speed well above the impossible-travel threshold (900 km/h) so
# the "impossible travel" signal is preserved, only the noise tail is tamed.
IMPLIED_SPEED_CAP_KMH = 5_000.0


def add_amount_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds ``amount_zscore``: how far this transaction's amount is from the
    customer's own historical mean, in units of the customer's own historical
    standard deviation -- computed with no lookahead.
    """
    df = df.copy()
    grp_amount = df.groupby("customer_id", group_keys=False)["amount_clp"]

    expanding_mean = grp_amount.apply(lambda s: s.shift(1).expanding().mean())
    expanding_std = grp_amount.apply(lambda s: s.shift(1).expanding().std())

    df["_hist_mean_amount"] = expanding_mean
    df["_hist_std_amount"] = expanding_std

    z = (df["amount_clp"] - df["_hist_mean_amount"]) / df["_hist_std_amount"]
    # Fewer than 2 prior transactions -> std is NaN/0 -> deviation undefined;
    # treat as "not yet deviating" rather than propagating NaN/inf.
    df["amount_zscore"] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df = df.drop(columns=["_hist_mean_amount", "_hist_std_amount"])
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Runs the full feature-engineering pipeline on raw transactions.

    Expects columns: transaction_id, customer_id, timestamp, amount_clp,
    merchant_category, latitude, longitude, (optional) is_fraud. Rows must be
    sorted by (customer_id, timestamp) -- ``generate_dataset`` already does
    this; re-sort defensively here so callers can't silently break the
    no-lookahead guarantees of the per-customer rolling features.
    """
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    df = add_calendar_features(df)
    df = add_time_velocity_features(df)
    df = add_geo_features(df)
    df = add_amount_deviation_features(df)

    df["seconds_since_prev"] = df["seconds_since_prev"].replace(
        np.inf, MAX_SECONDS_SINCE_PREV
    )
    # avg_seconds_between_txn is only used to derive velocity_ratio upstream
    # and would otherwise carry its own inf/NaN for first/second transactions.
    if "avg_seconds_between_txn" in df.columns:
        df = df.drop(columns=["avg_seconds_between_txn"])

    df["amount_zscore"] = df["amount_zscore"].clip(-AMOUNT_ZSCORE_CAP, AMOUNT_ZSCORE_CAP)
    df["velocity_ratio"] = df["velocity_ratio"].clip(0.0, VELOCITY_RATIO_CAP)
    df["implied_speed_kmh"] = df["implied_speed_kmh"].clip(upper=IMPLIED_SPEED_CAP_KMH)

    return df


if __name__ == "__main__":
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    raw_path = root / "data" / "raw" / "transactions.parquet"
    out_path = root / "data" / "processed" / "features.parquet"

    raw = pd.read_parquet(raw_path)
    features = build_feature_matrix(raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    print(f"Built features for {len(features):,} rows -> {out_path}")
    print(features[NUMERIC_FEATURE_COLUMNS + ["is_fraud"]].describe().T)
