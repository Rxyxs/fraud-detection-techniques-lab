import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anomaly.transaction_scorer import (
    FEATURE_COLUMNS_TX,
    build_transaction_features,
    evaluate_tx_against_ground_truth,
    run_transaction_ensemble,
    train_production_model,
)

UMBRAL = 5_000_000


def _synthetic_transfers(n_normal=300, n_pitufeo=15, seed=0):
    """Simula 'pitufeo': muchas transferencias sub-umbral desde la misma
    cuenta origen en una ventana corta de tiempo, marcadas es_ilicito=1."""
    rng = np.random.default_rng(seed)
    base = datetime(2026, 1, 1)

    filas = []
    for i in range(n_normal):
        filas.append({
            "transfer_id": f"TEF{i:06d}",
            "origen": f"CTA{rng.integers(0, 50):04d}",
            "destino": f"CTA{rng.integers(0, 50):04d}",
            "monto_clp": float(rng.uniform(50_000, 2_000_000)),
            "timestamp": base + timedelta(hours=int(rng.integers(0, 24 * 30))),
            "tipologia": "normal",
            "es_ilicito": 0,
        })

    t0 = base + timedelta(days=100)
    for j in range(n_pitufeo):
        filas.append({
            "transfer_id": f"TEFP{j:06d}",
            "origen": "CTA9999",
            "destino": "CTA9998",
            "monto_clp": float(4_900_000 + rng.uniform(-50_000, 90_000)),
            "timestamp": t0 + timedelta(minutes=10 * j),
            "tipologia": "pitufeo",
            "es_ilicito": 1,
        })

    return pl.DataFrame(filas)


def _account_features_stub(transfers: pl.DataFrame) -> pl.DataFrame:
    """Tabla de contexto por cuenta minima, con las columnas que
    ``build_transaction_features`` espera de ``graph_features.build_feature_table``."""
    origenes = transfers["origen"].unique().to_list()
    return pl.DataFrame({
        "account_id": origenes,
        "pagerank": [1.0 / len(origenes)] * len(origenes),
        "ratio_paso": [0.5] * len(origenes),
        "centralidad_intermediacion": [0.01] * len(origenes),
        "monto_prom_enviado": [500_000.0] * len(origenes),
        "monto_std_enviado": [200_000.0] * len(origenes),
        "monto_max_enviado": [2_000_000.0] * len(origenes),
        "antiguedad_dias": [365] * len(origenes),
    })


def test_build_transaction_features_has_expected_columns():
    transfers = _synthetic_transfers()
    features = build_transaction_features(transfers, _account_features_stub(transfers), UMBRAL)
    for col in FEATURE_COLUMNS_TX:
        assert col in features.columns
    assert features.height == transfers.height
    assert features.select(FEATURE_COLUMNS_TX).null_count().sum_horizontal().item() == 0


def test_pitufeo_transactions_get_high_pair_burst_count():
    transfers = _synthetic_transfers()
    features = build_transaction_features(transfers, _account_features_stub(transfers), UMBRAL)
    pitufeo_rows = features.sort("timestamp").filter(pl.col("tipologia") == "pitufeo")
    conteos = pitufeo_rows["n_par_24h"].to_list()
    # conteo causal: la primera transferencia del par solo se ve a si misma,
    # y el conteo crece a medida que llegan mas transferencias del mismo par
    # dentro de la ventana movil de 24h (sin fuga de informacion futura)
    assert conteos[0] == 1
    assert conteos[-1] == len(conteos)
    assert conteos == sorted(conteos)


def test_run_transaction_ensemble_flags_injected_pitufeo():
    transfers = _synthetic_transfers()
    features = build_transaction_features(transfers, _account_features_stub(transfers), UMBRAL)
    scored = run_transaction_ensemble(features, contamination=0.05)
    assert "alerta_tx" in scored.columns
    alertadas = scored.filter(pl.col("alerta_tx"))
    # la mayoria de las alertas emitidas deberian corresponder a las
    # transferencias de pitufeo inyectadas, no al ruido normal
    assert (alertadas["tipologia"] == "pitufeo").mean() > 0.5


def test_evaluate_tx_against_ground_truth_reasonable_auc():
    transfers = _synthetic_transfers()
    features = build_transaction_features(transfers, _account_features_stub(transfers), UMBRAL)
    scored = run_transaction_ensemble(features, contamination=0.05)
    metricas = evaluate_tx_against_ground_truth(scored)
    assert metricas["n_transacciones"] == transfers.height
    assert metricas["roc_auc_tx"] > 0.7


def test_train_production_model_persists_artifacts(tmp_path):
    transfers = _synthetic_transfers()
    features = build_transaction_features(transfers, _account_features_stub(transfers), UMBRAL)
    info = train_production_model(features, tmp_path, contamination=0.05)
    assert (tmp_path / "isolation_forest_produccion.joblib").exists()
    assert (tmp_path / "scaler_produccion.joblib").exists()
    assert (tmp_path / "metadata_produccion.joblib").exists()
    assert info["n_features"] == len(FEATURE_COLUMNS_TX)
