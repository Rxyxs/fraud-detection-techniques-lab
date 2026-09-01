"""Corre y compara los 3 enfoques de modelado complementarios sobre el mismo
dataset/pipeline (nivel transaccion):

    (a) baseline estadistico interpretable  (deep_baseline.statistical_baseline_score)
    (b) ensamble de arboles (IForest+COPOD+ECOD)  (transaction_scorer.run_transaction_ensemble)
    (c) autoencoder PyTorch, 3 activaciones (ReLU/GELU/Swish)  (deep_baseline.autoencoder_score)

Reusa los mismos datos sinteticos y features que ``src/pipeline.py`` (no
regenera nada); requiere haber corrido `python -m src.pipeline` al menos una
vez para que existan ``data/synthetic/*.parquet``. Persiste el comparativo en
DuckDB (``outputs/model_comparison.duckdb``) y genera las figuras
comparativas en ``outputs/figures/``.

Uso:
    python -m src.anomaly.run_model_comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.synthetic_generator import END_DATE, OUT_DIR as DATA_DIR, UMBRAL_ESTRUCTURACION_CLP
from src.anomaly.deep_baseline import ACTIVATIONS, autoencoder_score, evaluate_score, statistical_baseline_score
from src.anomaly.model_comparison_store import persist_comparison
from src.anomaly.transaction_scorer import build_transaction_features, run_transaction_ensemble
from src.graph.graph_features import build_feature_table
from src.graph.network_builder import build_weighted_digraph, ensure_all_accounts_present
from src.visualization.comparacion_modelos import (
    figura_activaciones_autoencoder,
    figura_comparacion_roc_auc,
    figura_curvas_roc,
)

OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
DB_PATH = OUTPUTS_DIR / "model_comparison.duckdb"

CONTAMINACION_TX = 0.01


def main(contamination: float = CONTAMINACION_TX, epochs: int = 60) -> dict:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    accounts = pl.read_parquet(DATA_DIR / "accounts.parquet")
    transfers = pl.read_parquet(DATA_DIR / "transfers.parquet")

    print("Reconstruyendo grafo y features (nivel cuenta y transaccion)...")
    graph = build_weighted_digraph(transfers)
    graph = ensure_all_accounts_present(graph, accounts["account_id"].to_list())
    account_features = build_feature_table(graph, transfers, accounts, UMBRAL_ESTRUCTURACION_CLP, END_DATE)
    tx_features = build_transaction_features(transfers, account_features, UMBRAL_ESTRUCTURACION_CLP)

    metricas_por_modelo: dict[str, dict] = {}
    curvas: dict[str, tuple] = {}

    print("(a) Baseline estadistico (z-score robusto MAD)...")
    scored_baseline = statistical_baseline_score(tx_features, contamination=contamination)
    m = evaluate_score(scored_baseline, "score_baseline", "alerta_baseline")
    metricas_por_modelo["baseline_estadistico"] = m
    curvas["baseline_estadistico"] = (scored_baseline["es_ilicito"].to_numpy(), scored_baseline["score_baseline"].to_numpy())
    print(f"    ROC-AUC={m['roc_auc']:.3f}  AP={m['average_precision']:.3f}")

    print("(b) Ensamble PyOD (IForest+COPOD+ECOD)...")
    scored_ensemble = run_transaction_ensemble(tx_features, contamination=contamination)
    m = evaluate_score(scored_ensemble, "score_ensamble_tx", "alerta_tx")
    metricas_por_modelo["ensamble_pyod"] = m
    curvas["ensamble_pyod"] = (scored_ensemble["es_ilicito"].to_numpy(), scored_ensemble["score_ensamble_tx"].to_numpy())
    print(f"    ROC-AUC={m['roc_auc']:.3f}  AP={m['average_precision']:.3f}")

    metricas_por_activacion: dict[str, dict] = {}
    for activacion in ACTIVATIONS:
        print(f"(c) Autoencoder PyTorch, activacion={activacion}...")
        scored_ae = autoencoder_score(tx_features, activation=activacion, contamination=contamination, epochs=epochs)
        score_col, alerta_col = f"score_autoencoder_{activacion}", f"alerta_autoencoder_{activacion}"
        m = evaluate_score(scored_ae, score_col, alerta_col)
        nombre = f"autoencoder_{activacion}"
        metricas_por_modelo[nombre] = m
        metricas_por_activacion[activacion] = m
        curvas[nombre] = (scored_ae["es_ilicito"].to_numpy(), scored_ae[score_col].to_numpy())
        print(f"    ROC-AUC={m['roc_auc']:.3f}  AP={m['average_precision']:.3f}")

    print(f"\nPersistiendo comparativo en {DB_PATH}...")
    persist_comparison(metricas_por_modelo, DB_PATH)

    print("Generando figuras comparativas...")
    figura_comparacion_roc_auc(metricas_por_modelo, FIGURES_DIR)
    figura_curvas_roc(curvas, FIGURES_DIR)
    figura_activaciones_autoencoder(metricas_por_activacion, FIGURES_DIR)

    print("\n=== COMPARATIVO DE MODELOS (nivel transaccion) ===")
    for nombre, m in sorted(metricas_por_modelo.items(), key=lambda kv: kv[1]["roc_auc"], reverse=True):
        print(f"  {nombre:24s} ROC-AUC={m['roc_auc']:.3f}  AP={m['average_precision']:.3f}  "
              f"precision={m['precision_en_alertas']:.3f}  recall={m['recall_sobre_verdad_terreno']:.3f}")

    return metricas_por_modelo


if __name__ == "__main__":
    main()
