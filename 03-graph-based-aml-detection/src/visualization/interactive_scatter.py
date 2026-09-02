"""Genera un scatter interactivo (Plotly, HTML autocontenido) de dos de las
features de grafo mas discriminantes del ensamble no supervisado -- PageRank
vs. ratio de paso -- coloreado por tipologia real (ground truth, usada solo
para colorear el punto, nunca para el ajuste del modelo) y con el tamanio del
punto proporcional al score del ensamble.

Requiere haber corrido antes `python -m src.pipeline` (deja
outputs/cuentas_con_score.parquet y data/synthetic/ground_truth.parquet).

Uso:
    python -m src.visualization.interactive_scatter
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import polars as pl
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "interactive"

TIPOLOGIA_COLORS = {
    "normal": "#8a8fa3",
    "pitufeo": "#e05252",
    "cuenta_puente": "#e0a052",
    "rafaga_cuenta_nueva": "#52a0e0",
    "monto_inusual": "#9b52e0",
}
TIPOLOGIA_LABELS = {
    "normal": "Normal",
    "pitufeo": "Structuring (pitufeo)",
    "cuenta_puente": "Bridge account",
    "rafaga_cuenta_nueva": "Fan-in burst, new account",
    "monto_inusual": "Unusual amount",
}


def main() -> None:
    cuentas = pl.read_parquet(ROOT / "outputs" / "cuentas_con_score.parquet")
    gt = pl.read_parquet(ROOT / "data" / "synthetic" / "ground_truth.parquet")

    df = cuentas.join(gt, on="account_id", how="left").with_columns(
        pl.col("tipologia_real").fill_null("normal")
    )

    fig = go.Figure()
    for tipologia, color in TIPOLOGIA_COLORS.items():
        sub = df.filter(pl.col("tipologia_real") == tipologia)
        if sub.height == 0:
            continue
        marker_size = 6 if tipologia == "normal" else 10
        fig.add_trace(
            go.Scatter(
                x=sub["pagerank"].to_list(),
                y=sub["ratio_paso"].to_list(),
                mode="markers",
                name=f"{TIPOLOGIA_LABELS[tipologia]} (n={sub.height})",
                marker=dict(
                    size=marker_size,
                    color=color,
                    opacity=0.55 if tipologia == "normal" else 0.85,
                    line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
                ),
                customdata=list(
                    zip(
                        sub["account_id"].to_list(),
                        sub["score_ensamble"].to_list(),
                        sub["banco"].to_list(),
                        sub["alerta"].to_list(),
                    )
                ),
                hovertemplate=(
                    "Account: %{customdata[0]}<br>"
                    "PageRank: %{x:.5f}<br>"
                    "Pass-through ratio: %{y:.3f}<br>"
                    "Ensemble score: %{customdata[1]:.3f}<br>"
                    "Bank: %{customdata[2]}<br>"
                    "Alert: %{customdata[3]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Graph features by AML typology (ground truth) — PageRank vs. pass-through ratio<br>"
        "<sup>2,000 accounts, unsupervised ensemble (IForest+COPOD+ECOD) never sees the typology label</sup>",
        xaxis_title="PageRank (importance in fund-flow structure)",
        yaxis_title="Pass-through ratio (min(in,out) / max(in,out) amount)",
        template="plotly_white",
        legend_title="Ground-truth typology",
        width=1000,
        height=650,
        hovermode="closest",
    )
    fig.update_xaxes(type="log")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "graph_features_scatter.html"
    fig.write_html(out_path, include_plotlyjs="inline", full_html=True)
    print(f"Guardado en: {out_path}")


if __name__ == "__main__":
    main()
