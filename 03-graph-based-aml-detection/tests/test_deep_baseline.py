import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anomaly.deep_baseline import (
    ACTIVATIONS,
    autoencoder_score,
    evaluate_score,
    statistical_baseline_score,
)
from src.anomaly.transaction_scorer import FEATURE_COLUMNS_TX

UMBRAL = 5_000_000


def _synthetic_tx_features(n_normal=200, n_anomalo=10, seed=0) -> pl.DataFrame:
    """Tabla minima con las columnas que ``deep_baseline`` espera: las de
    ``FEATURE_COLUMNS_TX`` mas ``es_ilicito``. Reusa el mismo espiritu que
    ``test_transaction_scorer._synthetic_transfers``: un grupo normal y un
    grupo con montos/frecuencias claramente extremas."""
    rng = np.random.default_rng(seed)
    base = datetime(2026, 1, 1)

    filas = []
    for i in range(n_normal):
        filas.append({
            "transfer_id": f"TEF{i:06d}",
            "origen": f"CTA{rng.integers(0, 50):04d}",
            "destino": f"CTA{rng.integers(0, 50):04d}",
            "monto_clp": float(rng.uniform(50_000, 1_000_000)),
            "timestamp": base + timedelta(hours=int(rng.integers(0, 24 * 30))),
            "tipologia": "normal",
            "es_ilicito": 0,
            "monto_log": float(np.log1p(rng.uniform(50_000, 1_000_000))),
            "ratio_a_umbral": float(rng.uniform(0.01, 0.3)),
            "cercano_umbral": 0.0,
            "hora": float(rng.integers(8, 20)),
            "es_nocturna": 0.0,
            "monto_zscore_origen": float(rng.normal(0, 1)),
            "monto_pct_max_origen": float(rng.uniform(0.1, 0.6)),
            "n_origen_24h": float(rng.integers(1, 5)),
            "n_par_24h": float(rng.integers(1, 3)),
            "dias_desde_apertura_origen": float(rng.integers(30, 1000)),
            "pagerank_origen": float(rng.uniform(0.0001, 0.001)),
            "ratio_paso_origen": float(rng.uniform(0, 0.3)),
            "centralidad_intermediacion_origen": float(rng.uniform(0, 0.01)),
        })

    for j in range(n_anomalo):
        filas.append({
            "transfer_id": f"TEFA{j:06d}",
            "origen": "CTA9999",
            "destino": "CTA9998",
            "monto_clp": float(4_950_000 + rng.uniform(-20_000, 40_000)),
            "timestamp": base + timedelta(days=100, minutes=10 * j),
            "tipologia": "pitufeo",
            "es_ilicito": 1,
            "monto_log": float(np.log1p(4_950_000)),
            "ratio_a_umbral": 0.99,
            "cercano_umbral": 1.0,
            "hora": 2.0,
            "es_nocturna": 1.0,
            "monto_zscore_origen": 8.0,
            "monto_pct_max_origen": 0.98,
            "n_origen_24h": 40.0,
            "n_par_24h": 40.0,
            "dias_desde_apertura_origen": 5.0,
            "pagerank_origen": 0.02,
            "ratio_paso_origen": 0.9,
            "centralidad_intermediacion_origen": 0.5,
        })

    return pl.DataFrame(filas).select(
        "transfer_id", "origen", "destino", "monto_clp", "timestamp", "tipologia", "es_ilicito",
        *FEATURE_COLUMNS_TX,
    )


def test_statistical_baseline_score_flags_expected_fraction():
    tx = _synthetic_tx_features()
    contamination = 0.05
    scored = statistical_baseline_score(tx, contamination=contamination)

    assert "score_baseline" in scored.columns
    assert "alerta_baseline" in scored.columns
    n_alertas = scored.filter(pl.col("alerta_baseline")).height
    esperado = int(np.ceil(contamination * tx.height))
    assert n_alertas == esperado


def test_statistical_baseline_score_ranks_anomalies_high():
    tx = _synthetic_tx_features()
    scored = statistical_baseline_score(tx, contamination=0.05)
    m = evaluate_score(scored, "score_baseline", "alerta_baseline")
    assert m["roc_auc"] > 0.7


@pytest.mark.parametrize("activation", list(ACTIVATIONS.keys()))
def test_autoencoder_score_runs_for_every_activation(activation):
    tx = _synthetic_tx_features(n_normal=80, n_anomalo=5)
    scored = autoencoder_score(tx, activation=activation, contamination=0.05, epochs=15)

    score_col, alerta_col = f"score_autoencoder_{activation}", f"alerta_autoencoder_{activation}"
    assert score_col in scored.columns
    assert alerta_col in scored.columns
    assert scored[score_col].null_count() == 0
    assert scored.filter(pl.col(alerta_col)).height >= 1


def test_autoencoder_score_rejects_unknown_activation():
    tx = _synthetic_tx_features(n_normal=20, n_anomalo=2)
    with pytest.raises(ValueError):
        autoencoder_score(tx, activation="tanh_not_supported", epochs=5)


def test_evaluate_score_returns_expected_keys():
    tx = _synthetic_tx_features()
    scored = statistical_baseline_score(tx, contamination=0.05)
    m = evaluate_score(scored, "score_baseline", "alerta_baseline")
    for key in (
        "roc_auc", "average_precision", "n_alertas", "n_alertas_correctas",
        "precision_en_alertas", "recall_sobre_verdad_terreno",
    ):
        assert key in m
