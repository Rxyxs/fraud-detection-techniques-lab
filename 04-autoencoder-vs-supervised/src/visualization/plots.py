"""Genera las figuras de resultados usadas en el README, a partir de los
artefactos reales dejados por el pipeline (requiere haber corrido antes
`python -m src.pipeline`). Ningun numero aqui es estimado: todo sale de
una corrida real.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "outputs" / "reports"
FIGURES_DIR = ROOT / "outputs" / "figures"

plt.style.use("seaborn-v0_8-darkgrid")
COLOR_AE = "#3498db"
COLOR_XGB = "#2ecc71"
COLOR_HYBRID = "#9b59b6"
COLOR_VAE = "#f39c12"
COLOR_SVDD = "#1abc9c"
COLOR_FRAUDE = "#e74c3c"
COLOR_NORMAL = "#3498db"


def figura_reconstruction_error():
    df = pd.read_parquet(REPORTS_DIR / "autoencoder_test_scores.parquet")
    normal = df.loc[df.y_true == 0, "reconstruction_error"]
    fraude = df.loc[df.y_true == 1, "reconstruction_error"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, np.percentile(df["reconstruction_error"], 99.5), 60)
    ax.hist(normal, bins=bins, alpha=0.7, label="Transacciones normales", color=COLOR_NORMAL, density=True)
    ax.hist(fraude, bins=bins, alpha=0.7, label="Fraude confirmado", color=COLOR_FRAUDE, density=True)
    ax.set_xlabel("Error de reconstruccion del autoencoder (MSE)")
    ax.set_ylabel("Densidad")
    ax.set_title("Autoencoder no supervisado: error de reconstruccion, normal vs. fraude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "reconstruction_error_distribution.png", dpi=150)
    plt.close(fig)


def figura_precision_recall_curves():
    ae = pd.read_parquet(REPORTS_DIR / "autoencoder_test_scores.parquet")
    xgb = pd.read_parquet(REPORTS_DIR / "supervised_test_scores.parquet")
    hybrid = pd.read_parquet(REPORTS_DIR / "hybrid_test_scores.parquet")

    series = [
        (ae, "reconstruction_error", COLOR_AE, "Autoencoder (no supervisado)"),
        (xgb, "xgb_score", COLOR_XGB, "XGBoost (supervisado)"),
        (hybrid, "hybrid_score", COLOR_HYBRID, "Hibrido (XGBoost + feature autoencoder)"),
    ]
    vae_path = REPORTS_DIR / "vae_test_scores.parquet"
    if vae_path.exists():
        series.append((pd.read_parquet(vae_path), "vae_score", COLOR_VAE, "VAE (no supervisado)"))
    deep_svdd_path = REPORTS_DIR / "deep_svdd_test_scores.parquet"
    if deep_svdd_path.exists():
        series.append((pd.read_parquet(deep_svdd_path), "deep_svdd_score", COLOR_SVDD, "Deep SVDD (no supervisado)"))

    fig, ax = plt.subplots(figsize=(8, 6))
    for df, col, color, label in series:
        precision, recall, _ = precision_recall_curve(df["y_true"], df[col])
        ax.plot(recall, precision, color=color, label=label, linewidth=2)

    baseline = df["y_true"].mean()
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, label=f"Azar (prevalencia={baseline:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall: no supervisado vs. supervisado vs. hibrido")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall_curves.png", dpi=150)
    plt.close(fig)


def figura_comparacion_pr_auc():
    import json
    with open(REPORTS_DIR / "comparison_summary.json") as f:
        resultados = json.load(f)

    nombres = {"autoencoder_no_supervisado": "Autoencoder\n(no supervisado)",
               "xgboost_supervisado": "XGBoost\n(supervisado)",
               "hibrido_xgboost_mas_autoencoder": "Hibrido\n(XGBoost + AE)",
               "vae_no_supervisado": "VAE\n(no supervisado)",
               "deep_svdd_no_supervisado": "Deep SVDD\n(no supervisado)"}
    colores_por_modelo = {"autoencoder_no_supervisado": COLOR_AE, "xgboost_supervisado": COLOR_XGB,
                           "hibrido_xgboost_mas_autoencoder": COLOR_HYBRID,
                           "vae_no_supervisado": COLOR_VAE, "deep_svdd_no_supervisado": COLOR_SVDD}
    labels = [nombres[m["nombre"]] for m in resultados["modelos"]]
    colores = [colores_por_modelo[m["nombre"]] for m in resultados["modelos"]]
    pr_aucs = [m["pr_auc"] for m in resultados["modelos"]]
    roc_aucs = [m["roc_auc"] for m in resultados["modelos"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(labels, pr_aucs, color=colores)
    axes[0].set_title("PR-AUC (average precision)")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(pr_aucs):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center")

    axes[1].bar(labels, roc_aucs, color=colores)
    axes[1].set_title("ROC-AUC")
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(roc_aucs):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center")

    fig.suptitle("Sin etiquetas vs. con etiquetas confirmadas: la brecha de desempenio real")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "comparacion_modelos.png", dpi=150)
    plt.close(fig)


def figura_shap_summary():
    shap_values = np.load(REPORTS_DIR / "shap_values_sample.npy")
    muestra = pd.read_parquet(REPORTS_DIR / "shap_sample_features.parquet")

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, muestra, show=False, plot_size=None)
    plt.title("SHAP: features que mas explican el score de fraude (XGBoost)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figura_reconstruction_error()
    figura_precision_recall_curves()
    figura_comparacion_pr_auc()
    figura_shap_summary()
    print(f"Figuras guardadas en {FIGURES_DIR}")


if __name__ == "__main__":
    main()
