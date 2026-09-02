"""Interactive ROC and PR curve comparison (Plotly, self-contained HTML)
across every model this project trains on the identical held-out test
split: the unsupervised autoencoder, VAE, and Deep SVDD (zero fraud labels
used), the supervised XGBoost baseline, and the hybrid (XGBoost +
autoencoder feature).

Requires having run `python -m src.pipeline` (and, for VAE/Deep SVDD,
`python -m src.models.train_vae` / `train_deep_svdd`) at least once, so the
`outputs/reports/*_test_scores.parquet` files exist.

Usage:
    python -m src.visualization.interactive_roc_pr
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "outputs" / "reports"
OUT_DIR = ROOT / "outputs" / "interactive"

MODELS = [
    ("autoencoder_test_scores.parquet", "reconstruction_error", "Autoencoder (unsupervised)", "#8a8fa3"),
    ("vae_test_scores.parquet", "vae_score", "VAE (unsupervised)", "#52a0e0"),
    ("deep_svdd_test_scores.parquet", "deep_svdd_score", "Deep SVDD (unsupervised)", "#9b52e0"),
    ("supervised_test_scores.parquet", "xgb_score", "XGBoost (supervised)", "#2e9e5b"),
    ("hybrid_test_scores.parquet", "hybrid_score", "Hybrid (XGBoost + AE)", "#e0a052"),
]


def main() -> None:
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("ROC curve", "Precision-Recall curve")
    )

    for filename, score_col, label, color in MODELS:
        path = REPORTS_DIR / filename
        if not path.exists():
            print(f"[skip] {path} not found, skipping {label}")
            continue
        df = pd.read_parquet(path)
        y_true = df["y_true"].to_numpy()
        y_score = df[score_col].to_numpy()

        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)

        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{label} (AUC={roc_auc:.3f})",
                line=dict(color=color, width=2), legendgroup=label,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=recall, y=precision, mode="lines", name=f"{label} (PR-AUC={pr_auc:.3f})",
                line=dict(color=color, width=2), legendgroup=label, showlegend=False,
            ),
            row=1, col=2,
        )

    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="lightgray", dash="dash"),
                    showlegend=False, hoverinfo="skip"),
        row=1, col=1,
    )

    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    fig.update_layout(
        title="Autoencoder vs. VAE vs. Deep SVDD vs. XGBoost vs. Hybrid — same held-out test set"
        "<br><sup>ROC-AUC is misleadingly generous at 0.172% fraud prevalence — PR-AUC is the metric that matters here</sup>",
        template="plotly_white",
        width=1200,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "roc_pr_comparison.html"
    fig.write_html(out_path, include_plotlyjs="inline", full_html=True)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
