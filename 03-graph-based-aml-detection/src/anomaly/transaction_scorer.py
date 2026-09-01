"""Scoring de anomalia a nivel de transaccion individual (no agregado por cuenta).

Motivacion (ver Seccion 7.3 del README): el ensamble por cuenta diluye dos
tipologias precisamente porque agrega toda la actividad de una cuenta en
estadisticos (media/std/max). Este modulo calcula un vector de features por
transaccion -- reusando el contexto de grafo/cuenta ya calculado en
``graph_features.build_feature_table`` mas senales especificas de la
transaccion -- y corre el mismo patron de ensamble PyOD no supervisado, esta
vez a granularidad de transferencia individual:

    - monto_zscore_origen / monto_pct_max_origen: que tan lejos esta ESTE
      monto del comportamiento propio de la cuenta origen. Ataca
      directamente el gap de "monto_inusual" que el modelo por cuenta pierde.
    - n_origen_24h / n_par_24h: conteo causal (solo transferencias pasadas,
      sin fuga de informacion) de transferencias en la ventana movil de 24h
      previas, por cuenta origen y por par origen-destino. Ataca el gap de
      "pitufeo" (fraccionamiento), donde cada transferencia individual no es
      extrema pero su frecuencia si lo es.

Ademas expone ``train_production_model``, que ajusta y persiste (joblib) un
``IsolationForest`` de scikit-learn -- no el ensamble PyOD completo -- porque
es el unico de los tres detectores que ``shap.TreeExplainer`` puede explicar
por transaccion en la API de scoring en vivo (ver ``app.py``).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import polars as pl
from pyod.models.combination import average
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.utils.utility import standardizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

NIGHT_HOURS = set(range(22, 24)) | set(range(0, 6))

FEATURE_COLUMNS_TX = [
    "monto_log",
    "ratio_a_umbral",
    "cercano_umbral",
    "hora",
    "es_nocturna",
    "monto_zscore_origen",
    "monto_pct_max_origen",
    "n_origen_24h",
    "n_par_24h",
    "dias_desde_apertura_origen",
    "pagerank_origen",
    "ratio_paso_origen",
    "centralidad_intermediacion_origen",
]

_EPS = 1e-6


def _rolling_causal_counts(transfers: pl.DataFrame) -> pl.DataFrame:
    """Conteo causal (solo pasado, ventana movil de 24h) de transferencias
    por cuenta origen y por par origen-destino, alineado de vuelta a cada
    fila via join. No usa informacion futura: la ventana de cada fila
    termina exactamente en su propio timestamp."""
    base = transfers.select("transfer_id", "origen", "destino", "timestamp").sort("timestamp").with_columns(
        (pl.col("origen") + "->" + pl.col("destino")).alias("par_id")
    )

    # ``rolling`` emite una fila por fila de entrada, no una por clave unica:
    # si dos transferencias comparten (grupo, timestamp) exacto, ambas
    # obtienen el mismo valor de ventana pero como filas separadas. Se
    # deduplica antes del join por clave para evitar una explosion
    # combinatoria (cada colision multiplicaria las filas resultantes).
    por_origen = (
        base.rolling(index_column="timestamp", period="24h", group_by="origen")
        .agg(pl.len().alias("n_origen_24h"))
        .unique(subset=["origen", "timestamp"], keep="first")
    )
    por_par = (
        base.rolling(index_column="timestamp", period="24h", group_by="par_id")
        .agg(pl.len().alias("n_par_24h"))
        .unique(subset=["par_id", "timestamp"], keep="first")
    )

    return (
        base.join(por_origen, on=["origen", "timestamp"], how="left")
        .join(por_par, on=["par_id", "timestamp"], how="left")
        .select("transfer_id", "n_origen_24h", "n_par_24h")
    )


def build_transaction_features(
    transfers: pl.DataFrame,
    account_features: pl.DataFrame,
    umbral_estructuracion: float,
) -> pl.DataFrame:
    """Construye la tabla de features por transaccion individual.

    ``account_features`` es la salida de ``graph_features.build_feature_table``
    (sin score): aporta el contexto de grafo/comportamiento historico de la
    cuenta origen (pagerank, ratio_paso, centralidad, monto_prom/std/max
    enviado) que cada transaccion individual hereda para contextualizar su
    propio monto.
    """
    contexto_origen = account_features.select(
        "account_id", "pagerank", "ratio_paso", "centralidad_intermediacion",
        "monto_prom_enviado", "monto_std_enviado", "monto_max_enviado",
        "antiguedad_dias",
    ).rename({
        "account_id": "origen",
        "pagerank": "pagerank_origen",
        "ratio_paso": "ratio_paso_origen",
        "centralidad_intermediacion": "centralidad_intermediacion_origen",
        "antiguedad_dias": "dias_desde_apertura_origen",
    })

    conteos = _rolling_causal_counts(transfers)

    features = (
        transfers.join(contexto_origen, on="origen", how="left")
        .join(conteos, on="transfer_id", how="left")
        .with_columns(
            pl.col("monto_clp").log1p().alias("monto_log"),
            (pl.col("monto_clp") / umbral_estructuracion).alias("ratio_a_umbral"),
            (
                (pl.col("monto_clp") >= 0.85 * umbral_estructuracion)
                & (pl.col("monto_clp") < umbral_estructuracion)
            ).cast(pl.Float64).alias("cercano_umbral"),
            pl.col("timestamp").dt.hour().alias("hora"),
            pl.col("timestamp").dt.hour().is_in(list(NIGHT_HOURS)).cast(pl.Float64).alias("es_nocturna"),
            (
                (pl.col("monto_clp") - pl.col("monto_prom_enviado"))
                / pl.col("monto_std_enviado").clip(lower_bound=_EPS)
            ).alias("monto_zscore_origen"),
            (
                pl.col("monto_clp") / pl.col("monto_max_enviado").clip(lower_bound=_EPS)
            ).alias("monto_pct_max_origen"),
        )
        .fill_null(0)
    )

    return features.select(
        "transfer_id", "origen", "destino", "monto_clp", "timestamp", "tipologia", "es_ilicito",
        *FEATURE_COLUMNS_TX,
    )


def _feature_matrix(tx_features: pl.DataFrame) -> np.ndarray:
    return tx_features.select(FEATURE_COLUMNS_TX).to_numpy().astype(float)


def run_transaction_ensemble(
    tx_features: pl.DataFrame,
    contamination: float = 0.01,
    random_state: int = 42,
) -> pl.DataFrame:
    """Mismo patron de ensamble no supervisado que ``ensemble_detector.run_ensemble``
    (IForest + COPOD + ECOD, promedio estandarizado), aplicado a nivel de
    transaccion individual en vez de cuenta agregada."""
    X = _feature_matrix(tx_features)

    detectores = {
        "iforest": IForest(contamination=contamination, random_state=random_state, n_estimators=200, n_jobs=1),
        "copod": COPOD(contamination=contamination),
        "ecod": ECOD(contamination=contamination),
    }

    scores_crudos = np.zeros((X.shape[0], len(detectores)))
    for i, (_, modelo) in enumerate(detectores.items()):
        modelo.fit(X)
        scores_crudos[:, i] = modelo.decision_scores_

    scores_std = standardizer(scores_crudos)
    score_promedio = average(scores_std)

    n_alertas = max(1, int(np.ceil(contamination * X.shape[0])))
    umbral = np.sort(score_promedio)[::-1][n_alertas - 1]

    return tx_features.with_columns(
        pl.Series("score_iforest_tx", scores_std[:, 0]),
        pl.Series("score_copod_tx", scores_std[:, 1]),
        pl.Series("score_ecod_tx", scores_std[:, 2]),
        pl.Series("score_ensamble_tx", score_promedio),
    ).with_columns(
        (pl.col("score_ensamble_tx") >= umbral).alias("alerta_tx")
    ).sort("score_ensamble_tx", descending=True)


def evaluate_tx_against_ground_truth(scored: pl.DataFrame) -> dict:
    """Evalua el ensamble de transacciones contra la etiqueta de verdad
    terreno que ya viene en cada fila de ``transfers`` (``es_ilicito``),
    inyectada por el generador y no usada para ajustar ningun detector."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = scored["es_ilicito"].to_numpy().astype(int)
    y_score = scored["score_ensamble_tx"].to_numpy()

    n_alertas_correctas = int(scored.filter(pl.col("alerta_tx") & (pl.col("es_ilicito") == 1)).height)
    n_alertas = int(scored.filter(pl.col("alerta_tx")).height)
    n_positivos = int(y_true.sum())

    return {
        "roc_auc_tx": float(roc_auc_score(y_true, y_score)) if 0 < n_positivos < len(y_true) else float("nan"),
        "average_precision_tx": float(average_precision_score(y_true, y_score)) if n_positivos > 0 else float("nan"),
        "n_transacciones": int(len(y_true)),
        "n_transacciones_ilicitas_verdad_terreno": n_positivos,
        "n_alertas_tx_emitidas": n_alertas,
        "n_alertas_tx_correctas": n_alertas_correctas,
        "precision_en_alertas_tx": (n_alertas_correctas / n_alertas) if n_alertas > 0 else float("nan"),
        "recall_sobre_verdad_terreno_tx": (n_alertas_correctas / n_positivos) if n_positivos > 0 else float("nan"),
    }


def train_production_model(
    tx_features: pl.DataFrame,
    model_dir: Path,
    contamination: float = 0.01,
    random_state: int = 42,
) -> dict:
    """Ajusta y persiste el modelo de scoring en vivo usado por ``app.py``.

    A diferencia de ``run_transaction_ensemble`` (ensamble PyOD de 3
    detectores, usado offline para el reporte de evaluacion), la API de
    produccion sirve un unico ``sklearn.ensemble.IsolationForest``: es el
    unico de los tres que ``shap.TreeExplainer`` puede explicar por
    transaccion (COPOD/ECOD no son modelos de arbol), y su costo de fit es
    bajo para reentrenar en cada corrida del pipeline.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    X = _feature_matrix(tx_features)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    modelo = IsolationForest(
        n_estimators=300, contamination=contamination, random_state=random_state, n_jobs=1
    ).fit(X_scaled)

    scores = -modelo.decision_function(X_scaled)  # mayor = mas anomalo
    n_alertas = max(1, int(np.ceil(contamination * X.shape[0])))
    umbral = np.sort(scores)[::-1][n_alertas - 1]

    joblib.dump(modelo, model_dir / "isolation_forest_produccion.joblib")
    joblib.dump(scaler, model_dir / "scaler_produccion.joblib")
    joblib.dump(
        {
            "feature_columns": FEATURE_COLUMNS_TX,
            "umbral_alerta": float(umbral),
            "contamination": contamination,
        },
        model_dir / "metadata_produccion.joblib",
    )
    return {"umbral_alerta": float(umbral), "n_features": len(FEATURE_COLUMNS_TX)}
