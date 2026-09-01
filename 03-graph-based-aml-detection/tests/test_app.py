import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as api_app
from src.anomaly.transaction_scorer import build_transaction_features, train_production_model
from src.api import store


@pytest.fixture()
def cliente_api(tmp_path, monkeypatch):
    """Entrena un modelo de produccion desechable sobre datos sinteticos
    pequenios y apunta la API a directorios temporales, en vez de depender
    de haber corrido `python -m src.pipeline` antes de `pytest`."""
    rng = np.random.default_rng(0)
    base = datetime(2026, 1, 1)
    origenes = [f"CTA{i:04d}" for i in range(20)]

    filas = []
    for i in range(200):
        filas.append({
            "transfer_id": f"TEF{i:06d}",
            "origen": rng.choice(origenes),
            "destino": rng.choice(origenes),
            "monto_clp": float(rng.uniform(50_000, 2_000_000)),
            "timestamp": base + timedelta(hours=int(rng.integers(0, 24 * 30))),
            "tipologia": "normal",
            "es_ilicito": 0,
        })
    transfers = pl.DataFrame(filas)

    cuentas = pl.DataFrame({
        "account_id": origenes,
        "pagerank": [1.0 / len(origenes)] * len(origenes),
        "ratio_paso": [0.5] * len(origenes),
        "centralidad_intermediacion": [0.01] * len(origenes),
        "monto_prom_enviado": [500_000.0] * len(origenes),
        "monto_std_enviado": [200_000.0] * len(origenes),
        "monto_max_enviado": [2_000_000.0] * len(origenes),
        "antiguedad_dias": [365] * len(origenes),
    })

    tx_features = build_transaction_features(transfers, cuentas, umbral_estructuracion=5_000_000)

    modelos_dir = tmp_path / "models"
    outputs_dir = tmp_path / "outputs"
    data_dir = tmp_path / "data"
    outputs_dir.mkdir()
    data_dir.mkdir()

    train_production_model(tx_features, modelos_dir, contamination=0.05)
    cuentas.write_parquet(outputs_dir / "cuentas_con_score.parquet")
    transfers.write_parquet(data_dir / "transfers.parquet")

    monkeypatch.setattr(api_app, "MODELS_DIR", modelos_dir)
    monkeypatch.setattr(api_app, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(api_app, "DATA_DIR", data_dir)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "decisiones_analista.db")

    api_app._cargar_estado()
    store.init_db(store.DB_PATH)

    return TestClient(api_app.app)


def test_health_reports_model_loaded(cliente_api):
    resp = cliente_api.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "modelo_cargado": True}


def test_score_returns_alert_flag_and_threshold(cliente_api):
    resp = cliente_api.post("/score", json={
        "origen": "CTA0001", "destino": "CTA0002", "monto_clp": 4_900_000,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"transfer_id", "score_anomalia", "umbral_alerta", "es_alerta"}
    assert isinstance(body["es_alerta"], bool)


def test_explicar_returns_shap_contributions_for_every_feature(cliente_api):
    from src.anomaly.transaction_scorer import FEATURE_COLUMNS_TX

    resp = cliente_api.post("/explicar", json={
        "origen": "CTA0001", "destino": "CTA0002", "monto_clp": 4_900_000,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["contribuciones"]) == len(FEATURE_COLUMNS_TX)
    features_devueltas = {c["feature"] for c in body["contribuciones"]}
    assert features_devueltas == set(FEATURE_COLUMNS_TX)


def test_decision_persiste_y_aparece_en_alertas(cliente_api):
    score_resp = cliente_api.post("/score", json={
        "origen": "CTA0001", "destino": "CTA0002", "monto_clp": 4_900_000,
    })
    transfer_id = score_resp.json()["transfer_id"]

    decision_resp = cliente_api.post("/decisiones", json={
        "transfer_id": transfer_id,
        "analista": "pablo.reyes",
        "decision": "confirmado_ilicito",
        "notas": "prueba",
    })
    assert decision_resp.status_code == 200

    alertas = cliente_api.get("/alertas").json()
    coincidencias = [a for a in alertas if a["transfer_id"] == transfer_id]
    if score_resp.json()["es_alerta"]:
        assert len(coincidencias) == 1
        assert coincidencias[0]["decision"] == "confirmado_ilicito"


def test_score_cuenta_desconocida_no_falla(cliente_api):
    resp = cliente_api.post("/score", json={
        "origen": "CTA_INEXISTENTE", "destino": "CTA0002", "monto_clp": 1_000_000,
    })
    assert resp.status_code == 200
