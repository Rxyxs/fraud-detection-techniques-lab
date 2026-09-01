"""API de produccion (FastAPI) del motor de deteccion de anomalias AML.

Sirve el modelo entrenado por ``python -m src.pipeline``
(``src/anomaly/transaction_scorer.train_production_model``) para:

    - Scoring en vivo de una transaccion individual (``POST /score``).
    - Explicabilidad SHAP por transaccion, feature a feature
      (``POST /explicar``).
    - Persistencia del historial de scoring y de las decisiones de un
      analista sobre cada alerta, en SQLite (``POST /decisiones``,
      ``GET /alertas``).

Requiere haber corrido antes ``python -m src.pipeline``, que deja los
artefactos usados aqui (modelo, scaler, features historicas de contexto) en
``models/`` y ``outputs/``.

Uso:
    uvicorn app:app --reload
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import polars as pl
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.anomaly.transaction_scorer import FEATURE_COLUMNS_TX
from src.api import store

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data" / "synthetic"
UMBRAL_ESTRUCTURACION_CLP = 5_000_000
_EPS = 1e-6

_estado: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _cargar_estado()
    store.init_db()
    yield


app = FastAPI(
    title="Motor de Deteccion de Anomalias AML - API de Produccion",
    description=(
        "Scoring de anomalia en vivo a nivel de transaccion individual, "
        "explicabilidad SHAP y persistencia de decisiones de analista."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _cargar_estado() -> None:
    modelo_path = MODELS_DIR / "isolation_forest_produccion.joblib"
    if not modelo_path.exists():
        _estado["modelo"] = None
        return

    _estado["modelo"] = joblib.load(modelo_path)
    _estado["scaler"] = joblib.load(MODELS_DIR / "scaler_produccion.joblib")
    _estado["metadata"] = joblib.load(MODELS_DIR / "metadata_produccion.joblib")
    _estado["explainer"] = shap.TreeExplainer(_estado["modelo"])

    cuentas = pl.read_parquet(OUTPUTS_DIR / "cuentas_con_score.parquet").select(
        "account_id", "pagerank", "ratio_paso", "centralidad_intermediacion",
        "monto_prom_enviado", "monto_std_enviado", "monto_max_enviado", "antiguedad_dias",
    )
    _estado["contexto_cuentas"] = {
        row["account_id"]: row for row in cuentas.iter_rows(named=True)
    }

    transfers = pl.read_parquet(DATA_DIR / "transfers.parquet").select("origen", "destino", "timestamp")
    _estado["transferencias_historicas"] = transfers


class TransaccionEntrada(BaseModel):
    transfer_id: Optional[str] = Field(None, description="Si se omite, se genera uno")
    origen: str = Field(..., description="account_id de la cuenta origen")
    destino: str = Field(..., description="account_id de la cuenta destino")
    monto_clp: float = Field(..., gt=0)
    timestamp: Optional[datetime] = Field(None, description="Si se omite, se usa la hora actual (UTC)")


class ScoreSalida(BaseModel):
    transfer_id: str
    score_anomalia: float
    umbral_alerta: float
    es_alerta: bool


class ContribucionFeature(BaseModel):
    feature: str
    valor: float
    contribucion_shap: float


class ExplicacionSalida(BaseModel):
    transfer_id: str
    score_anomalia: float
    es_alerta: bool
    valor_base_shap: float
    contribuciones: list[ContribucionFeature]


class DecisionEntrada(BaseModel):
    transfer_id: str
    analista: str
    decision: str = Field(..., pattern="^(confirmado_ilicito|falso_positivo|pendiente_revision)$")
    notas: Optional[str] = None


def _requiere_modelo() -> None:
    if _estado.get("modelo") is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo de produccion no encontrado. Ejecuta `python -m src.pipeline` primero.",
        )


def _construir_features(tx: TransaccionEntrada) -> tuple[str, np.ndarray]:
    """Replica, para una transaccion nueva en vivo, el mismo vector de
    features que ``transaction_scorer.build_transaction_features`` calcula
    offline sobre el historico -- incluyendo los conteos causales de 24h,
    que aqui se resuelven filtrando el historico cargado en memoria en vez
    de una ventana rolling batch."""
    transfer_id = tx.transfer_id or f"LIVE{int(datetime.now(timezone.utc).timestamp() * 1000)}"
    ts = tx.timestamp or datetime.now(timezone.utc)
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)

    contexto = _estado["contexto_cuentas"].get(tx.origen)
    if contexto is None:
        pagerank = ratio_paso = centralidad = monto_prom = monto_std = monto_max = antiguedad = 0.0
    else:
        pagerank = contexto["pagerank"]
        ratio_paso = contexto["ratio_paso"]
        centralidad = contexto["centralidad_intermediacion"]
        monto_prom = contexto["monto_prom_enviado"]
        monto_std = contexto["monto_std_enviado"]
        monto_max = contexto["monto_max_enviado"]
        antiguedad = contexto["antiguedad_dias"]

    historicas = _estado["transferencias_historicas"]
    ventana_inicio = ts - timedelta(hours=24)
    n_origen_24h = historicas.filter(
        (pl.col("origen") == tx.origen)
        & (pl.col("timestamp") > ventana_inicio)
        & (pl.col("timestamp") <= ts)
    ).height + 1  # +1: incluye la transaccion actual, aun no en el historico
    n_par_24h = historicas.filter(
        (pl.col("origen") == tx.origen) & (pl.col("destino") == tx.destino)
        & (pl.col("timestamp") > ventana_inicio)
        & (pl.col("timestamp") <= ts)
    ).height + 1

    monto_std_seguro = max(monto_std, _EPS)
    monto_max_seguro = max(monto_max, _EPS)
    hora = ts.hour
    es_nocturna = 1.0 if (hora >= 22 or hora < 6) else 0.0
    cercano_umbral = 1.0 if (0.85 * UMBRAL_ESTRUCTURACION_CLP <= tx.monto_clp < UMBRAL_ESTRUCTURACION_CLP) else 0.0

    valores = {
        "monto_log": float(np.log1p(tx.monto_clp)),
        "ratio_a_umbral": tx.monto_clp / UMBRAL_ESTRUCTURACION_CLP,
        "cercano_umbral": cercano_umbral,
        "hora": float(hora),
        "es_nocturna": es_nocturna,
        "monto_zscore_origen": (tx.monto_clp - monto_prom) / monto_std_seguro,
        "monto_pct_max_origen": tx.monto_clp / monto_max_seguro,
        "n_origen_24h": float(n_origen_24h),
        "n_par_24h": float(n_par_24h),
        "dias_desde_apertura_origen": float(antiguedad),
        "pagerank_origen": float(pagerank),
        "ratio_paso_origen": float(ratio_paso),
        "centralidad_intermediacion_origen": float(centralidad),
    }
    X = np.array([[valores[col] for col in FEATURE_COLUMNS_TX]], dtype=float)
    return transfer_id, X, valores


def _score_vector(X: np.ndarray) -> tuple[float, bool]:
    X_scaled = _estado["scaler"].transform(X)
    score = float(-_estado["modelo"].decision_function(X_scaled)[0])
    umbral = _estado["metadata"]["umbral_alerta"]
    return score, score >= umbral


@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": _estado.get("modelo") is not None}


@app.post("/score", response_model=ScoreSalida)
def score(tx: TransaccionEntrada):
    _requiere_modelo()
    transfer_id, X, _ = _construir_features(tx)
    puntaje, es_alerta = _score_vector(X)

    store.guardar_scoring(
        transfer_id=transfer_id,
        origen=tx.origen,
        destino=tx.destino,
        monto_clp=tx.monto_clp,
        timestamp_transaccion=(tx.timestamp or datetime.now(timezone.utc)).isoformat(),
        score_anomalia=puntaje,
        es_alerta=es_alerta,
    )

    return ScoreSalida(
        transfer_id=transfer_id,
        score_anomalia=puntaje,
        umbral_alerta=_estado["metadata"]["umbral_alerta"],
        es_alerta=es_alerta,
    )


@app.post("/explicar", response_model=ExplicacionSalida)
def explicar(tx: TransaccionEntrada):
    _requiere_modelo()
    transfer_id, X, valores = _construir_features(tx)
    X_scaled = _estado["scaler"].transform(X)
    puntaje, es_alerta = _score_vector(X)

    shap_values = _estado["explainer"].shap_values(X_scaled)
    contribuciones = sorted(
        [
            ContribucionFeature(feature=col, valor=valores[col], contribucion_shap=float(shap_values[0][i]))
            for i, col in enumerate(FEATURE_COLUMNS_TX)
        ],
        key=lambda c: abs(c.contribucion_shap),
        reverse=True,
    )

    return ExplicacionSalida(
        transfer_id=transfer_id,
        score_anomalia=puntaje,
        es_alerta=es_alerta,
        valor_base_shap=float(np.ravel(_estado["explainer"].expected_value)[0]),
        contribuciones=contribuciones,
    )


@app.post("/decisiones")
def registrar_decision(entrada: DecisionEntrada):
    store.guardar_decision(
        transfer_id=entrada.transfer_id,
        analista=entrada.analista,
        decision=entrada.decision,
        notas=entrada.notas,
    )
    return {"status": "registrado"}


@app.get("/alertas")
def listar_alertas(limite: int = 100):
    return store.listar_alertas(limite=limite)
