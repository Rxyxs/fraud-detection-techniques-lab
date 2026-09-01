"""Graficos del enfoque de deep learning (MLP + Focal Loss), siguiendo la
paleta y estilo ya usados en el notebook de EDA (seaborn whitegrid,
mismos colores por clase/modelo) para que los reportes se vean como un
solo sistema visual."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "outputs" / "reports"

# Misma paleta que 01_Fraud_EDA_and_Model_Comparison.ipynb
COLOR_ACTIVATION = {"relu": "#4C72B0", "gelu": "#DD8452", "swish": "#55A868"}
COLOR_MODEL = {
    "logreg_smote": "#8C8C8C",
    "catboost": "#4C72B0",
    "xgboost": "#DD8452",
    "mlp_focal_loss": "#937860",
}


def plot_loss_curves(history: pd.DataFrame, filename: str = "mlp_loss_curves.png") -> Path:
    """Curvas de loss (train y validacion) por epoca, una linea por
    funcion de activacion (ReLU/GELU/Swish) sobre el mismo split."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for activation, color in COLOR_ACTIVATION.items():
        sub = history[history["activation"] == activation]
        axes[0].plot(sub["epoch"], sub["train_loss"], color=color, label=activation, marker="o", markersize=3)
        axes[1].plot(sub["epoch"], sub["val_loss"], color=color, label=activation, marker="o", markersize=3)

    axes[0].set_title("Focal Loss (train) por epoca")
    axes[0].set_xlabel("epoca")
    axes[0].set_ylabel("focal loss")
    axes[0].legend()

    axes[1].set_title("Focal Loss (validacion) por epoca")
    axes[1].set_xlabel("epoca")
    axes[1].set_ylabel("focal loss")
    axes[1].legend()

    plt.tight_layout()
    out_path = REPORTS_DIR / filename
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_loss_curves_animated(
    history: pd.DataFrame, filename: str = "mlp_loss_curves_animated.gif"
) -> Path:
    """Version 'racing line chart' de plot_loss_curves: la curva de
    validation loss por epoca se dibuja progresivamente para cada
    activacion (ReLU/GELU/Swish), usando los mismos datos reales de
    mlp_activation_history.csv (sin fabricar valores)."""
    plt.style.use("dark_background")

    series = {}
    max_epoch = 0
    for activation in COLOR_ACTIVATION:
        sub = history[history["activation"] == activation].sort_values("epoch")
        series[activation] = (sub["epoch"].to_numpy(), sub["val_loss"].to_numpy())
        max_epoch = max(max_epoch, len(sub))

    n_frames = min(50, max(max_epoch, 2))
    # Subsample real epoch indices to n_frames steps (never fabricate values).
    frame_steps = np.unique(
        np.linspace(1, max_epoch, num=n_frames, dtype=int)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    lines = {}
    labels = {}
    for activation, color in COLOR_ACTIVATION.items():
        (line,) = ax.plot([], [], color=color, label=activation, linewidth=2)
        lines[activation] = line
        labels[activation] = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 0),
            textcoords="offset points",
            color="black",
            fontsize=9,
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none", alpha=0.9),
        )

    all_epochs = np.concatenate([s[0] for s in series.values() if len(s[0])])
    all_losses = np.concatenate([s[1] for s in series.values() if len(s[1])])
    ax.set_xlim(all_epochs.min(), all_epochs.max())
    ax.set_ylim(0, all_losses.max() * 1.15)
    ax.set_xlabel("epoca")
    ax.set_ylabel("focal loss (validacion)")
    ax.set_title("Focal Loss (validacion) por epoca — ReLU vs GELU vs Swish")
    ax.legend(loc="upper right")

    def update(frame_idx):
        upto = frame_steps[frame_idx]
        for activation, (epochs, losses) in series.items():
            mask = epochs <= upto
            xs, ys = epochs[mask], losses[mask]
            lines[activation].set_data(xs, ys)
            if len(xs):
                labels[activation].xy = (xs[-1], ys[-1])
                labels[activation].set_text(f"{activation}: {ys[-1]:.4f}")
            else:
                labels[activation].set_text("")
        return list(lines.values()) + list(labels.values())

    ani = FuncAnimation(fig, update, frames=len(frame_steps), interval=150, blit=False)

    out_path = REPORTS_DIR / filename
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ani.save(out_path, writer="pillow")
    plt.close(fig)
    plt.style.use("default")
    return out_path


def plot_model_comparison_with_mlp(metrics: pd.DataFrame, filename: str = "model_comparison_with_mlp.png") -> Path:
    """Igual que la celda 'model_comparison.png' del notebook pero
    incluyendo el 4to modelo (MLP + Focal Loss), para comparar PR-AUC y
    reduccion de costo de negocio entre los 4 enfoques."""
    sns.set_theme(style="whitegrid")
    colors = [COLOR_MODEL.get(m, "#333333") for m in metrics["model"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(metrics["model"], metrics["pr_auc"], color=colors)
    axes[0].set_title("PR-AUC por modelo (4 enfoques)")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(metrics["model"], metrics["cost_reduction_pct"], color=colors)
    axes[1].set_title("Reduccion de costo de negocio vs. umbral 0.5")
    axes[1].set_ylabel("% reduccion")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    out_path = REPORTS_DIR / filename
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
