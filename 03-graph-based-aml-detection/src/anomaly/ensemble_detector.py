"""Ensamble de deteccion de anomalias no supervisado (PyOD).

Combina tres detectores complementarios sobre la misma tabla de features
por cuenta (metricas de grafo + transaccionales):

    - IsolationForest : particiona el espacio de features; efectivo en
                        anomalias de combinacion multivariada (p.ej. alta
                        centralidad + ratio de paso alto + cuenta nueva).
    - COPOD           : "Copula-Based Outlier Detection", no parametrico,
                        rapido y efectivo en colas pesadas (montos extremos).
    - ECOD            : "Empirical CDF", sin hiperparametros, robusto para
                        outliers marginales por variable (p.ej. un monto
                        aislado muy por sobre el resto de una columna).

Cada detector se entrena sin ninguna etiqueta. Los scores se estandarizan
(z-score) para hacerlos comparables entre si y luego se combinan con
``pyod.models.combination.average``. La columna ``es_ilicito``/``tipologia``
de verdad terreno NUNCA se usa como input del modelo: solo se usa despues,
en la evaluacion, para medir que tan bien el ensamble no supervisado
recupera los casos inyectados (estandar en benchmarks de deteccion de
fraude/LA con datos sinteticos).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from pyod.models.combination import average, maximization
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.utils.utility import standardizer

FEATURE_COLUMNS = [
    "grado_entrada", "grado_salida", "monto_entrada_total", "monto_salida_total",
    "pagerank", "coef_agrupamiento", "centralidad_intermediacion", "reciprocidad_local",
    "tamanio_componente_conexa", "ratio_paso",
    "n_enviadas", "monto_prom_enviado", "monto_std_enviado", "monto_max_enviado",
    "n_cercanas_umbral", "n_recibidas", "monto_prom_recibido", "monto_std_recibido",
    "monto_max_recibido", "burst_score_24h", "antiguedad_dias", "ratio_nocturno",
]


def _feature_matrix(features: pl.DataFrame) -> np.ndarray:
    return features.select(FEATURE_COLUMNS).to_numpy().astype(float)


def run_ensemble(
    features: pl.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pl.DataFrame:
    """Ajusta el ensamble no supervisado y retorna ``features`` con columnas
    de score agregadas: ``score_iforest``, ``score_copod``, ``score_ecod``,
    ``score_ensamble`` (promedio estandarizado) y ``alerta`` (top-N segun
    ``contamination``)."""
    X = _feature_matrix(features)

    detectores = {
        "iforest": IForest(contamination=contamination, random_state=random_state, n_estimators=200, n_jobs=1),
        "copod": COPOD(contamination=contamination),
        "ecod": ECOD(contamination=contamination),
    }

    scores_crudos = np.zeros((X.shape[0], len(detectores)))
    for i, (nombre, modelo) in enumerate(detectores.items()):
        modelo.fit(X)
        scores_crudos[:, i] = modelo.decision_scores_

    scores_std = standardizer(scores_crudos)
    score_promedio = average(scores_std)
    score_maximo = maximization(scores_std)

    n_alertas = max(1, int(np.ceil(contamination * X.shape[0])))
    umbral = np.sort(score_promedio)[::-1][n_alertas - 1]

    return features.with_columns(
        pl.Series("score_iforest", scores_std[:, 0]),
        pl.Series("score_copod", scores_std[:, 1]),
        pl.Series("score_ecod", scores_std[:, 2]),
        pl.Series("score_ensamble", score_promedio),
        pl.Series("score_ensamble_max", score_maximo),
    ).with_columns(
        (pl.col("score_ensamble") >= umbral).alias("alerta")
    ).sort("score_ensamble", descending=True)


def evaluate_against_ground_truth(scored_features: pl.DataFrame, ground_truth: pl.DataFrame) -> dict:
    """Evalua el ensamble no supervisado contra la verdad terreno sintetica.

    Uso exclusivo de validacion metodologica: en produccion no existiria
    esta etiqueta. Retorna ROC-AUC, average precision y precision/recall
    del conjunto de alertas emitido."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    etiquetas = ground_truth.select("account_id").with_columns(pl.lit(1).alias("es_ilicito_real"))
    df = scored_features.join(etiquetas, on="account_id", how="left").with_columns(
        pl.col("es_ilicito_real").fill_null(0)
    )

    y_true = df["es_ilicito_real"].to_numpy()
    y_score = df["score_ensamble"].to_numpy()

    n_pos_en_alerta = int(df.filter(pl.col("alerta") & (pl.col("es_ilicito_real") == 1)).height)
    n_alertas = int(df.filter(pl.col("alerta")).height)
    n_positivos = int(y_true.sum())

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if 0 < n_positivos < len(y_true) else float("nan"),
        "average_precision": float(average_precision_score(y_true, y_score)) if n_positivos > 0 else float("nan"),
        "n_cuentas": int(len(y_true)),
        "n_cuentas_ilicitas_verdad_terreno": n_positivos,
        "n_alertas_emitidas": n_alertas,
        "n_alertas_correctas": n_pos_en_alerta,
        "precision_en_alertas": (n_pos_en_alerta / n_alertas) if n_alertas > 0 else float("nan"),
        "recall_sobre_verdad_terreno": (n_pos_en_alerta / n_positivos) if n_positivos > 0 else float("nan"),
    }
