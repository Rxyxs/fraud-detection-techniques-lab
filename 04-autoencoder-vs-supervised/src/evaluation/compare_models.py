"""Compara, sobre el MISMO test set nunca visto por ningun modelo durante
ajuste o calibracion, tres enfoques:

    1. Autoencoder no supervisado (entrenado solo con transacciones
       normales) — el escenario realista de un problema de fraude/LA sin
       ninguna etiqueta confirmada todavia.
    2. XGBoost supervisado (entrenado con las etiquetas reales) — el
       escenario de una institucion con historial de fraude confirmado.
    3. Hibrido: XGBoost supervisado + el error de reconstruccion del
       autoencoder como feature adicional.

Uso:
    python -m src.evaluation.compare_models
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "outputs" / "reports"

ALERT_BUDGETS = [20, 50, 100, 200, 500, 1000]


def _metrics_at_budget(y_true: np.ndarray, y_score: np.ndarray, k: int) -> dict:
    orden = np.argsort(-y_score)
    top_k = orden[:k]
    tp = int(y_true[top_k].sum())
    n_positivos = int(y_true.sum())
    return {
        "n_alertas": k,
        "tp": tp,
        "precision": tp / k,
        "recall": tp / n_positivos if n_positivos > 0 else float("nan"),
    }


def evaluar(nombre: str, y_true: np.ndarray, y_score: np.ndarray) -> dict:
    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    sweep = [_metrics_at_budget(y_true, y_score, k) for k in ALERT_BUDGETS]
    return {"nombre": nombre, "roc_auc": roc_auc, "pr_auc": pr_auc, "sweep_por_presupuesto": sweep}


def main():
    ae = pd.read_parquet(REPORTS_DIR / "autoencoder_test_scores.parquet")
    xgb = pd.read_parquet(REPORTS_DIR / "supervised_test_scores.parquet")
    hybrid = pd.read_parquet(REPORTS_DIR / "hybrid_test_scores.parquet")

    assert (ae["y_true"].to_numpy() == xgb["y_true"].to_numpy()).all(), "Los splits de test no coinciden"
    assert (ae["y_true"].to_numpy() == hybrid["y_true"].to_numpy()).all(), "Los splits de test no coinciden"
    y_true = ae["y_true"].to_numpy()
    n_test = len(y_true)
    n_fraude = int(y_true.sum())

    modelos = [
        evaluar("autoencoder_no_supervisado", y_true, ae["reconstruction_error"].to_numpy()),
        evaluar("xgboost_supervisado", y_true, xgb["xgb_score"].to_numpy()),
        evaluar("hibrido_xgboost_mas_autoencoder", y_true, hybrid["hybrid_score"].to_numpy()),
    ]

    vae_path = REPORTS_DIR / "vae_test_scores.parquet"
    if vae_path.exists():
        vae = pd.read_parquet(vae_path)
        assert (vae["y_true"].to_numpy() == y_true).all(), "Los splits de test no coinciden"
        modelos.append(evaluar("vae_no_supervisado", y_true, vae["vae_score"].to_numpy()))

    deep_svdd_path = REPORTS_DIR / "deep_svdd_test_scores.parquet"
    if deep_svdd_path.exists():
        deep_svdd = pd.read_parquet(deep_svdd_path)
        assert (deep_svdd["y_true"].to_numpy() == y_true).all(), "Los splits de test no coinciden"
        modelos.append(evaluar("deep_svdd_no_supervisado", y_true, deep_svdd["deep_svdd_score"].to_numpy()))

    resultados = {
        "n_test": n_test,
        "n_fraude_test": n_fraude,
        "modelos": modelos,
    }

    with open(REPORTS_DIR / "comparison_summary.json", "w") as f:
        json.dump(resultados, f, indent=2)

    print(f"Test set: {n_test:,} transacciones ({n_fraude} fraudes)\n")
    for modelo in resultados["modelos"]:
        print(f"=== {modelo['nombre']} ===")
        print(f"  ROC-AUC: {modelo['roc_auc']:.4f} | PR-AUC: {modelo['pr_auc']:.4f}")
        for punto in modelo["sweep_por_presupuesto"]:
            print(f"  top-{punto['n_alertas']:<5} precision={punto['precision']:.3f}  recall={punto['recall']:.3f}")
        print()

    reporte_md = REPORTS_DIR / "comparison_report.md"
    with open(reporte_md, "w", encoding="utf-8") as f:
        f.write("# Comparacion: autoencoder no supervisado vs. XGBoost supervisado vs. hibrido\n\n")
        f.write(f"Test set: {n_test:,} transacciones reales, nunca vistas por ningun modelo ({n_fraude} fraudes confirmados).\n\n")
        f.write("| Modelo | ROC-AUC | PR-AUC |\n|---|---:|---:|\n")
        for modelo in resultados["modelos"]:
            f.write(f"| {modelo['nombre']} | {modelo['roc_auc']:.4f} | {modelo['pr_auc']:.4f} |\n")
        f.write("\n## Precision/recall por presupuesto de alertas\n\n")
        for modelo in resultados["modelos"]:
            f.write(f"\n**{modelo['nombre']}**\n\n| Alertas | Precision | Recall |\n|---:|---:|---:|\n")
            for punto in modelo["sweep_por_presupuesto"]:
                f.write(f"| {punto['n_alertas']} | {punto['precision']:.3f} | {punto['recall']:.3f} |\n")

    print(f"Reporte escrito en {reporte_md}")
    return resultados


if __name__ == "__main__":
    main()
