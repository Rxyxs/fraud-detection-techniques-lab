"""Genera las figuras estaticas usadas en el README a partir de los
artefactos reales de ``outputs/`` (requiere haber corrido antes
``python -m src.pipeline``). No genera cifras estimadas: todo se calcula
sobre el resultado real de la ultima corrida del pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.synthetic_generator import END_DATE, OUT_DIR as DATA_DIR, UMBRAL_ESTRUCTURACION_CLP
from src.anomaly.ensemble_detector import evaluate_against_ground_truth, run_ensemble
from src.graph.graph_features import build_feature_table
from src.graph.network_builder import build_weighted_digraph, ensure_all_accounts_present

OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

plt.style.use("seaborn-v0_8-darkgrid")
COLOR_NORMAL = "#3498db"
COLOR_ILICITO = "#e74c3c"


def figura_distribucion_scores(cuentas: pl.DataFrame, ground_truth: pl.DataFrame):
    df = cuentas.join(
        ground_truth.select("account_id").with_columns(pl.lit(True).alias("es_ilicita")),
        on="account_id", how="left",
    ).with_columns(pl.col("es_ilicita").fill_null(False))

    normales = df.filter(~pl.col("es_ilicita"))["score_ensamble"].to_numpy()
    ilicitas = df.filter(pl.col("es_ilicita"))["score_ensamble"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(normales, bins=40, alpha=0.7, label="Cuentas normales", color=COLOR_NORMAL, density=True)
    ax.hist(ilicitas, bins=40, alpha=0.7, label="Cuentas de tipologia AML (verdad terreno)", color=COLOR_ILICITO, density=True)
    ax.set_xlabel("Score de anomalia del ensamble (IForest + COPOD + ECOD, estandarizado)")
    ax.set_ylabel("Densidad")
    ax.set_title("Separacion de scores: cuentas normales vs. cuentas con tipologia AML inyectada")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distribucion_scores.png", dpi=150)
    plt.close(fig)


def figura_precision_recall_sweep(features: pl.DataFrame, ground_truth: pl.DataFrame):
    niveles = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    precisiones, recalls = [], []
    for c in niveles:
        scored = run_ensemble(features, contamination=c)
        m = evaluate_against_ground_truth(scored, ground_truth)
        precisiones.append(m["precision_en_alertas"])
        recalls.append(m["recall_sobre_verdad_terreno"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(niveles, precisiones, marker="o", label="Precision en alertas", color=COLOR_ILICITO)
    ax.plot(niveles, recalls, marker="s", label="Recall sobre verdad terreno", color=COLOR_NORMAL)
    ax.set_xlabel("Contaminacion (fraccion de cuentas marcadas como alerta)")
    ax.set_ylabel("Metrica")
    ax.set_title("Trade-off precision/recall segun presupuesto de alertas")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall_sweep.png", dpi=150)
    plt.close(fig)
    return niveles, precisiones, recalls


def figura_precision_recall_sweep_animada(niveles, precisiones, recalls):
    """Version 'racing line chart' del sweep precision/recall: las lineas se
    dibujan progresivamente cuadro a cuadro sobre los mismos datos reales
    (``niveles``/``precisiones``/``recalls``) ya calculados por
    ``figura_precision_recall_sweep``. Estilo oscuro solo para esta figura,
    la version estatica no se toca."""
    x = np.array(niveles)
    y_prec = np.array(precisiones)
    y_rec = np.array(recalls)
    n = len(x)
    n_frames = max(n, min(45, n * 6))

    # Sub-muestreo de indices reales (sin inventar valores) para tener
    # suficientes cuadros de animacion aunque haya pocos niveles reales.
    frame_positions = np.linspace(0, n - 1, n_frames)

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Contaminacion (fraccion de cuentas marcadas como alerta)")
        ax.set_ylabel("Metrica")
        ax.set_title("Trade-off precision/recall segun presupuesto de alertas (animado)")

        line_prec, = ax.plot([], [], marker="o", color=COLOR_ILICITO, label="Precision en alertas", linewidth=2)
        line_rec, = ax.plot([], [], marker="s", color=COLOR_NORMAL, label="Recall sobre verdad terreno", linewidth=2)
        ax.legend(loc="upper right")

        label_prec = ax.annotate(
            "", xy=(0, 0), xytext=(15, 10), textcoords="offset points",
            color="white", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=COLOR_ILICITO, ec="none", alpha=0.9),
        )
        label_rec = ax.annotate(
            "", xy=(0, 0), xytext=(15, -20), textcoords="offset points",
            color="white", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=COLOR_NORMAL, ec="none", alpha=0.9),
        )

        def update(frame_idx):
            pos = frame_positions[frame_idx]
            k = int(np.floor(pos))
            frac = pos - k
            if k >= n - 1:
                idx = n
                xi, yi_p, yi_r = x[-1], y_prec[-1], y_rec[-1]
            else:
                idx = k + 1
                xi = x[k] + frac * (x[k + 1] - x[k])
                yi_p = y_prec[k] + frac * (y_prec[k + 1] - y_prec[k])
                yi_r = y_rec[k] + frac * (y_rec[k + 1] - y_rec[k])

            line_prec.set_data(np.append(x[:idx], xi), np.append(y_prec[:idx], yi_p))
            line_rec.set_data(np.append(x[:idx], xi), np.append(y_rec[:idx], yi_r))

            label_prec.xy = (xi, yi_p)
            label_prec.set_text(f"Precision: {yi_p:.3f}")
            label_rec.xy = (xi, yi_r)
            label_rec.set_text(f"Recall: {yi_r:.3f}")
            return line_prec, line_rec, label_prec, label_rec

        ani = FuncAnimation(fig, update, frames=n_frames, interval=120, blit=False)
        ani.save(FIGURES_DIR / "precision_recall_sweep_animated.gif", writer="pillow")
        plt.close(fig)


def figura_tipologias_en_alertas(alertas: pl.DataFrame):
    dist = (
        alertas.with_columns(pl.col("tipologia_real").fill_null("sin_tipologia_conocida"))
        .group_by("tipologia_real").agg(pl.len().alias("n")).sort("n", descending=True)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    colores = {
        "pitufeo": "#e74c3c", "cuenta_puente": "#e67e22", "rafaga_cuenta_nueva": "#9b59b6",
        "monto_inusual": "#f1c40f", "sin_tipologia_conocida": "#95a5a6",
    }
    ax.bar(
        dist["tipologia_real"].to_list(), dist["n"].to_list(),
        color=[colores.get(t, "#7f8c8d") for t in dist["tipologia_real"].to_list()],
    )
    ax.set_ylabel("N cuentas en alerta")
    ax.set_title("Tipologia real de las cuentas marcadas como alerta por el ensamble")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tipologias_en_alertas.png", dpi=150)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    accounts = pl.read_parquet(DATA_DIR / "accounts.parquet")
    transfers = pl.read_parquet(DATA_DIR / "transfers.parquet")
    ground_truth = pl.read_parquet(DATA_DIR / "ground_truth.parquet")
    cuentas = pl.read_parquet(OUTPUTS_DIR / "cuentas_con_score.parquet")
    alertas = pl.read_csv(OUTPUTS_DIR / "alertas_uaf.csv")

    graph = build_weighted_digraph(transfers)
    graph = ensure_all_accounts_present(graph, accounts["account_id"].to_list())
    features = build_feature_table(graph, transfers, accounts, UMBRAL_ESTRUCTURACION_CLP, END_DATE)

    figura_distribucion_scores(cuentas, ground_truth)
    niveles, precisiones, recalls = figura_precision_recall_sweep(features, ground_truth)
    figura_precision_recall_sweep_animada(niveles, precisiones, recalls)
    figura_tipologias_en_alertas(alertas)

    print("Figuras guardadas en", FIGURES_DIR)
    print("\nSweep precision/recall (para tabla del README):")
    for c, p, r in zip(niveles, precisiones, recalls):
        n_alertas = round(c * features.height)
        print(f"  contaminacion={c:<5} n_alertas={n_alertas:4d}  precision={p:.3f}  recall={r:.3f}")


if __name__ == "__main__":
    main()
