"""FastAPI real-time fraud-scoring service.

Latency budget: < 50ms per request (see ``tests/test_latency.py``). To hit
that under high concurrency, this endpoint deliberately does NOT perform any
historical database lookup or heavy feature recomputation per request -- see
``src/api/schemas.py`` for why ``customer_state`` is passed in by the caller
(the online feature store) instead. What's left on the hot path is O(1)
feature arithmetic in plain numpy plus two already-loaded, CPU-bound
tree-model inferences, which is comfortably sub-50ms even in-process.
"""
from __future__ import annotations

import pathlib
import time
from contextlib import asynccontextmanager
from datetime import datetime

import joblib
import numpy as np
import torch
from catboost import CatBoostClassifier
from fastapi import FastAPI
from xgboost import XGBClassifier

from src.api.schemas import FraudDetectionResponse, HealthResponse, TransactionRequest
from src.features.build_features import (
    AMOUNT_ZSCORE_CAP,
    IMPLIED_SPEED_CAP_KMH,
    MAX_SECONDS_SINCE_PREV,
    NUMERIC_FEATURE_COLUMNS,
    VELOCITY_RATIO_CAP,
)
from src.features.geo_distance import IMPOSSIBLE_TRAVEL_KMH, haversine_km
from src.models.autoencoder import FraudAutoencoder, reconstruction_error

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "outputs" / "models"

FULL_FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + ["autoencoder_score"]

_state: dict = {}


def _load_artifacts() -> dict:
    import json

    with open(MODELS_DIR / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    scaler = joblib.load(MODELS_DIR / "scaler.joblib")

    ae_model = FraudAutoencoder(
        n_features=metadata["autoencoder_n_features"],
        latent_dim=metadata["autoencoder_latent_dim"],
    )
    ae_model.load_state_dict(torch.load(MODELS_DIR / "autoencoder.pt", map_location="cpu"))
    ae_model.eval()

    cat_model = CatBoostClassifier()
    cat_model.load_model(str(MODELS_DIR / "catboost_fraud.cbm"))

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(MODELS_DIR / "xgboost_fraud.json"))

    return {
        "metadata": metadata,
        "scaler": scaler,
        "autoencoder": ae_model,
        "catboost": cat_model,
        "xgboost": xgb_model,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state.update(_load_artifacts())
    yield
    _state.clear()


app = FastAPI(
    title="Chile Financial Fraud Detection API",
    description="Real-time fraud scoring for Chilean e-commerce / card payments.",
    version="1.0.0",
    lifespan=lifespan,
)


def _build_feature_vector(req: TransactionRequest) -> np.ndarray:
    txn_ts = datetime.fromisoformat(req.timestamp)
    cs = req.customer_state

    hour_of_day = txn_ts.hour
    day_of_week = txn_ts.weekday()

    if cs.last_timestamp is not None:
        last_ts = datetime.fromisoformat(cs.last_timestamp)
        seconds_since_prev = max((txn_ts - last_ts).total_seconds(), 0.0)
    else:
        seconds_since_prev = MAX_SECONDS_SINCE_PREV

    if cs.avg_seconds_between_txn and cs.avg_seconds_between_txn > 0:
        velocity_ratio = seconds_since_prev / cs.avg_seconds_between_txn
    else:
        velocity_ratio = 1.0
    velocity_ratio = float(np.clip(velocity_ratio, 0.0, VELOCITY_RATIO_CAP))

    if cs.last_latitude is not None and cs.last_longitude is not None:
        distance_from_prev_km = float(
            haversine_km(req.latitude, req.longitude, cs.last_latitude, cs.last_longitude)
        )
    else:
        distance_from_prev_km = 0.0

    if seconds_since_prev > 0:
        implied_speed_kmh = distance_from_prev_km / (seconds_since_prev / 3600.0)
    elif distance_from_prev_km > 0:
        implied_speed_kmh = IMPOSSIBLE_TRAVEL_KMH * 10  # simultaneous, different place
    else:
        implied_speed_kmh = 0.0
    implied_speed_kmh = float(min(implied_speed_kmh, IMPLIED_SPEED_CAP_KMH))
    is_impossible_travel = 1.0 if implied_speed_kmh > IMPOSSIBLE_TRAVEL_KMH else 0.0

    if cs.home_latitude is not None and cs.home_longitude is not None:
        distance_from_home_km = float(
            haversine_km(req.latitude, req.longitude, cs.home_latitude, cs.home_longitude)
        )
    else:
        distance_from_home_km = 0.0

    if cs.avg_amount_clp is not None and cs.std_amount_clp and cs.std_amount_clp > 0:
        amount_zscore = (req.amount_clp - cs.avg_amount_clp) / cs.std_amount_clp
    else:
        amount_zscore = 0.0
    amount_zscore = float(np.clip(amount_zscore, -AMOUNT_ZSCORE_CAP, AMOUNT_ZSCORE_CAP))

    values = {
        "amount_clp": req.amount_clp,
        "amount_zscore": amount_zscore,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "seconds_since_prev": seconds_since_prev,
        "velocity_ratio": velocity_ratio,
        "txn_count_last_1h": cs.txn_count_last_1h,
        "txn_count_last_24h": cs.txn_count_last_24h,
        "amount_sum_last_1h": cs.amount_sum_last_1h,
        "distance_from_prev_km": distance_from_prev_km,
        "implied_speed_kmh": implied_speed_kmh,
        "is_impossible_travel": is_impossible_travel,
        "distance_from_home_km": distance_from_home_km,
    }
    return np.array([[values[c] for c in NUMERIC_FEATURE_COLUMNS]], dtype=np.float64)


def _explain(feature_row: dict) -> list[str]:
    reasons = []
    if feature_row["is_impossible_travel"]:
        reasons.append(
            f"Impossible travel: {feature_row['implied_speed_kmh']:.0f} km/h implied speed "
            f"from the previous transaction location."
        )
    if feature_row["txn_count_last_1h"] >= 3:
        reasons.append(
            f"High velocity: {int(feature_row['txn_count_last_1h'])} other transactions "
            f"in the last hour."
        )
    if feature_row["amount_zscore"] >= 3:
        reasons.append(
            f"Unusual amount: {feature_row['amount_zscore']:.1f} standard deviations "
            f"above this customer's historical average."
        )
    if not reasons:
        reasons.append("No individual high-risk signal crossed its alert threshold.")
    return reasons


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _state else "loading",
        model_version=str(_state.get("metadata", {}).get("seed", "unknown")),
        feature_columns=FULL_FEATURE_COLUMNS,
    )


@app.post("/detect-fraud", response_model=FraudDetectionResponse)
def detect_fraud(req: TransactionRequest) -> FraudDetectionResponse:
    start = time.perf_counter()

    numeric_vector = _build_feature_vector(req)
    feature_row = dict(zip(NUMERIC_FEATURE_COLUMNS, numeric_vector[0].tolist()))

    scaled = _state["scaler"].transform(numeric_vector)
    ae_score = float(reconstruction_error(_state["autoencoder"], scaled)[0])

    full_vector = np.concatenate([numeric_vector, np.array([[ae_score]])], axis=1)

    proba_cat = _state["catboost"].predict_proba(full_vector)[0, 1]
    proba_xgb = _state["xgboost"].predict_proba(full_vector)[0, 1]
    fraud_probability = float((proba_cat + proba_xgb) / 2.0)

    threshold = _state["metadata"]["decision_threshold"]
    is_fraud = fraud_probability >= threshold

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return FraudDetectionResponse(
        transaction_id=req.transaction_id,
        fraud_probability=fraud_probability,
        is_fraud=is_fraud,
        decision_threshold=threshold,
        model_latency_ms=elapsed_ms,
        reasons=_explain(feature_row),
    )
