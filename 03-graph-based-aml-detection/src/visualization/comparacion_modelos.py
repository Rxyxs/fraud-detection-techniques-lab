"""Figuras del comparativo de enfoques de modelado (baseline estadistico vs.
ensamble PyOD vs. autoencoder PyTorch por activacion). Mismo estilo que
``generar_figuras_reporte.py`` (paleta y ``plt.style`` compartidos) para que
las figuras se vean consistentes entre si en el README."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import roc_curve

plt.style.use("seaborn-v0_8-darkgrid")
COLOR_NORMAL = "#3498db"
COLOR_ILICITO = "#e74c3c"
PALETA_MODELOS = {
    "baseline_estadistico": "#95a5a6",
    "ensamble_pyod": "#3498db",
    "autoencoder_relu": "#e67e22",
    "autoencoder_gelu": "#9b59b6",
    "autoencoder_swish": "#e74c3c",
}


def figura_comparacion_roc_auc(metricas_por_modelo: dict[str, dict], figures_dir: Path):
    """Barras de ROC-AUC por modelo -- resume el comparativo completo en una
    sola figura, ordenado de mejor a peor desempenio."""
    modelos = sorted(metricas_por_modelo, key=lambda m: metricas_por_modelo[m]["roc_auc"], reverse=True)
    valores = [metricas_por_modelo[m]["roc_auc"] for m in modelos]
    colores = [PALETA_MODELOS.get(m, "#7f8c8d") for m in modelos]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(modelos, valores, color=colores)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1)
    ax.set_title("Comparacion de enfoques de modelado (nivel transaccion)")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Azar (AUC=0.5)")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(figures_dir / "comparacion_modelos_auc.png", dpi=150)
    plt.close(fig)


def figura_curvas_roc(scored_by_model: dict[str, tuple[np.ndarray, np.ndarray]], figures_dir: Path):
    """Curvas ROC superpuestas de cada modelo, dado ``{nombre: (y_true, y_score)}``."""
    fig, ax = plt.subplots(figsize=(7, 7))
    for nombre, (y_true, y_score) in scored_by_model.items():
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, label=nombre, color=PALETA_MODELOS.get(nombre, None), linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.4, label="Azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC -- comparacion de enfoques (nivel transaccion)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "comparacion_modelos_roc.png", dpi=150)
    plt.close(fig)


def figura_activaciones_autoencoder(metricas_por_activacion: dict[str, dict], figures_dir: Path):
    """Compara ROC-AUC y average precision del autoencoder segun activacion
    (ReLU / GELU / Swish), misma arquitectura y epochs para las tres."""
    activaciones = list(metricas_por_activacion.keys())
    roc_aucs = [metricas_por_activacion[a]["roc_auc"] for a in activaciones]
    aps = [metricas_por_activacion[a]["average_precision"] for a in activaciones]

    x = np.arange(len(activaciones))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, roc_aucs, width, label="ROC-AUC", color=COLOR_NORMAL)
    ax.bar(x + width / 2, aps, width, label="Average precision", color=COLOR_ILICITO)
    ax.set_xticks(x)
    ax.set_xticklabels(activaciones)
    ax.set_ylim(0, 1)
    ax.set_title("Autoencoder PyTorch: ReLU vs. GELU vs. Swish (misma arquitectura)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "comparacion_activaciones_autoencoder.png", dpi=150)
    plt.close(fig)
