"""Interactive Plotly chart (self-contained HTML) of XGBoost's predicted
fraud probability distribution on the real held-out test set, split by true
class -- the same real predictions persisted to SQLite by `src.pipeline`
(`outputs/fraud.sqlite`, table `predictions`).

Requires having run `python -m src.pipeline` at least once.

Usage:
    python -m src.interactive_plots
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "interactive"


def main() -> None:
    con = sqlite3.connect(ROOT / "outputs" / "fraud.sqlite")
    df = pd.read_sql("SELECT y_true, proba_xgboost FROM predictions", con)
    con.close()

    legit = df.loc[df["y_true"] == 0, "proba_xgboost"]
    fraud = df.loc[df["y_true"] == 1, "proba_xgboost"]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=legit, name=f"Legitimate (n={len(legit):,})", marker_color="#3b7ddd",
            opacity=0.75, nbinsx=60,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=fraud, name=f"Fraud (n={len(fraud):,})", marker_color="#c0392b",
            opacity=0.75, nbinsx=60,
        )
    )
    fig.add_vline(
        x=0.7337, line_dash="dash", line_color="#2e2e2e",
        annotation_text="cost-optimal threshold (0.734)", annotation_position="top",
    )
    fig.update_layout(
        barmode="overlay",
        yaxis_type="log",
        title="XGBoost predicted fraud probability — real held-out test set (113,726 transactions)"
        "<br><sup>568,629 real 2023 card-fraud transactions, y-axis log-scaled to show the small overlap region</sup>",
        xaxis_title="Predicted P(fraud)",
        yaxis_title="Transactions (log scale)",
        template="plotly_white",
        width=1000,
        height=600,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fraud_probability_distribution.html"
    fig.write_html(out_path, include_plotlyjs="inline", full_html=True)
    print(f"Guardado en: {out_path}")


if __name__ == "__main__":
    main()
