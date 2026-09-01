"""End-to-end training pipeline:

raw transactions -> feature engineering -> time-based split -> autoencoder
anomaly score -> SMOTE+Tomek on the train split -> cost-sensitive
CatBoost/XGBoost -> threshold tuned on the validation split to minimize
business cost -> final evaluation on the held-out test split -> artifacts
saved to ``outputs/``.

A time-based (not random) split is used deliberately: shuffling transactions
across time would let the model "see the future" via a customer's own later
transactions during training, which does not reflect how a real-time fraud
system is deployed.
"""
from __future__ import annotations

import json
import pathlib

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler

from src.data.generate_transactions import time_based_split
from src.features.build_features import NUMERIC_FEATURE_COLUMNS, build_feature_matrix
from src.models.autoencoder import reconstruction_error, train_autoencoder
from src.models.catboost_fraud import (
    compute_sample_weights,
    evaluate,
    find_optimal_threshold,
    resample_train_split,
    train_models,
)
from src.models.logistic_baseline import (
    coefficient_report,
    predict_proba as logistic_predict_proba,
    train_logistic_baseline,
)
from src.models.mlp_focal import compare_activations
from src.models.mlp_focal import predict_proba as mlp_predict_proba
from src import metrics_store

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "transactions.parquet"
PROCESSED_PATH = ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = ROOT / "outputs" / "models"
PLOTS_DIR = ROOT / "outputs" / "plots"
REPORTS_DIR = ROOT / "outputs" / "reports"

FULL_FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + ["autoencoder_score"]
SEED = 42


def _add_autoencoder_score(train_df, val_df, test_df, scaler, ae_model, device):
    for split_df in (train_df, val_df, test_df):
        X_scaled = scaler.transform(split_df[NUMERIC_FEATURE_COLUMNS].to_numpy())
        split_df["autoencoder_score"] = reconstruction_error(ae_model, X_scaled, device=device)
    return train_df, val_df, test_df


def _save_plots(y_test, y_proba, threshold, feature_importance: dict):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#c0392b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Test Set)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "precision_recall_curve.png", dpi=150)
    plt.close()

    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Reds")
    plt.title(f"Confusion Matrix (threshold={threshold:.3f})")
    plt.xticks([0, 1], ["Legit", "Fraud"])
    plt.yticks([0, 1], ["Legit", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=12)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    names = list(feature_importance.keys())
    values = list(feature_importance.values())
    order = np.argsort(values)
    plt.figure(figsize=(7, 6))
    plt.barh([names[i] for i in order], [values[i] for i in order], color="#2c3e50")
    plt.xlabel("CatBoost feature importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=150)
    plt.close()


def _save_dl_loss_curves(histories: dict):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    colors = {"relu": "#2c3e50", "gelu": "#c0392b", "swish": "#27ae60"}
    for activation, history in histories.items():
        epochs = [h["epoch"] for h in history]
        val_losses = [h["val_loss"] for h in history]
        plt.plot(epochs, val_losses, label=activation, color=colors.get(activation))
    plt.xlabel("Epoch")
    plt.ylabel("Validation Focal Loss")
    plt.title("MLP Validation Loss by Activation (Focal Loss)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "dl_loss_curves.png", dpi=150)
    plt.close()


def _save_dl_loss_curves_animated(histories: dict):
    """Racing line-chart animation of the same MLP validation-loss curves
    saved statically by ``_save_dl_loss_curves``. Uses the exact same
    per-epoch history data — no synthetic values."""
    import matplotlib.animation as animation

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"relu": "#00d4ff", "gelu": "#ff5c8a", "swish": "#7dff8a"}

    with plt.style.context("dark_background"):
        n_frames = min(60, max(len(h) for h in histories.values()))
        fig, ax = plt.subplots(figsize=(12, 6))

        all_epochs = [h["epoch"] for h in next(iter(histories.values()))]
        all_losses = [v for h in histories.values() for pt in h for v in (pt["val_loss"],)]
        x_min, x_max = min(all_epochs), max(all_epochs)
        y_min, y_max = min(all_losses), max(all_losses)
        y_pad = (y_max - y_min) * 0.15 or 0.01

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(max(0, y_min - y_pad), y_max + y_pad)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Focal Loss")
        ax.set_title("MLP Validation Loss by Activation (Focal Loss)")
        ax.grid(alpha=0.2)

        lines = {}
        labels = {}
        for activation in histories:
            (line,) = ax.plot([], [], label=activation, color=colors.get(activation, "#ffffff"), linewidth=2)
            lines[activation] = line
            labels[activation] = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(10, 0),
                textcoords="offset points",
                color="black",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc=colors.get(activation, "#ffffff"), ec="none", alpha=0.9),
                va="center",
            )
        ax.legend(loc="upper right")

        def update(frame):
            idx = int(round((frame + 1) / n_frames * len(all_epochs)))
            idx = max(1, min(idx, len(all_epochs)))
            for activation, history in histories.items():
                epochs = [h["epoch"] for h in history[:idx]]
                val_losses = [h["val_loss"] for h in history[:idx]]
                lines[activation].set_data(epochs, val_losses)
                if epochs:
                    labels[activation].set_position((10, 0))
                    labels[activation].xy = (epochs[-1], val_losses[-1])
                    labels[activation].set_text(f"{activation}: {val_losses[-1]:.4f}")
            return list(lines.values()) + list(labels.values())

        ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=120, blit=False)
        ani.save(PLOTS_DIR / "dl_loss_curves_animated.gif", writer="pillow")
        plt.close(fig)


def _save_model_comparison_plot(comparison_rows: list[dict]):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    names = [row["model_name"] for row in comparison_rows]
    aucs = [row["roc_auc"] for row in comparison_rows]
    order = np.argsort(aucs)
    plt.figure(figsize=(7, 5))
    plt.barh([names[i] for i in order], [aucs[i] for i in order], color="#8e44ad")
    plt.xlabel("ROC-AUC (test set)")
    plt.title("Model Comparison: Baseline vs. Tree Ensemble vs. Deep Learning")
    plt.xlim(min(aucs) - 0.01 if aucs else 0, 1.0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150)
    plt.close()


def main():
    print("[1/7] Loading raw transactions...")
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_PATH} not found. Run `python -m src.data.generate_transactions` first."
        )
    raw = pd.read_parquet(RAW_PATH)

    print("[2/7] Building features...")
    features = build_feature_matrix(raw)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(PROCESSED_PATH, index=False)

    print("[3/7] Time-based train/val/test split...")
    train_df, val_df, test_df = time_based_split(features)
    print(
        f"  train={len(train_df):,} (fraud={train_df['is_fraud'].sum()}), "
        f"val={len(val_df):,} (fraud={val_df['is_fraud'].sum()}), "
        f"test={len(test_df):,} (fraud={test_df['is_fraud'].sum()})"
    )

    print("[4/7] Training autoencoder on legit training transactions...")
    scaler = StandardScaler().fit(train_df[NUMERIC_FEATURE_COLUMNS].to_numpy())
    X_train_scaled = scaler.transform(train_df[NUMERIC_FEATURE_COLUMNS].to_numpy())
    legit_mask = (train_df["is_fraud"] == 0).to_numpy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae_artifacts = train_autoencoder(
        X_train_scaled[legit_mask], n_features=len(NUMERIC_FEATURE_COLUMNS), device=device, seed=SEED
    )
    print(f"  final autoencoder train loss: {ae_artifacts.train_losses[-1]:.5f}")

    train_df, val_df, test_df = _add_autoencoder_score(
        train_df, val_df, test_df, scaler, ae_artifacts.model, device
    )

    print("[5/7] SMOTE+Tomek resampling (train split only) + cost-sensitive fit...")
    X_train = train_df[FULL_FEATURE_COLUMNS]
    y_train = train_df["is_fraud"].to_numpy()
    X_train_res, y_train_res = resample_train_split(X_train, y_train, seed=SEED)
    print(f"  train rows before={len(X_train):,}, after SMOTE+Tomek={len(X_train_res):,}")

    sample_weight_res = compute_sample_weights(
        y_train_res, X_train_res["amount_clp"].to_numpy()
    )
    models = train_models(X_train_res, y_train_res, sample_weight_train=sample_weight_res, seed=SEED)

    print("[6/7] Tuning decision threshold on validation split (minimize business cost)...")
    X_val = val_df[FULL_FEATURE_COLUMNS]
    y_val = val_df["is_fraud"].to_numpy()
    val_proba_cat = models.catboost.predict_proba(X_val)[:, 1]
    val_proba_xgb = models.xgboost.predict_proba(X_val)[:, 1]
    val_proba_ensemble = (val_proba_cat + val_proba_xgb) / 2.0

    threshold, val_cost = find_optimal_threshold(
        y_val, val_proba_ensemble, val_df["amount_clp"].to_numpy()
    )
    print(f"  optimal threshold={threshold:.3f}, validation cost=CLP {val_cost:,.0f}")

    print("[7/7] Final evaluation on held-out test split...")
    X_test = test_df[FULL_FEATURE_COLUMNS]
    y_test = test_df["is_fraud"].to_numpy()
    test_proba_cat = models.catboost.predict_proba(X_test)[:, 1]
    test_proba_xgb = models.xgboost.predict_proba(X_test)[:, 1]
    test_proba_ensemble = (test_proba_cat + test_proba_xgb) / 2.0

    metrics_ensemble = evaluate(y_test, test_proba_ensemble, test_df["amount_clp"].to_numpy(), threshold)
    metrics_catboost = evaluate(y_test, test_proba_cat, test_df["amount_clp"].to_numpy(), threshold)
    metrics_xgboost = evaluate(y_test, test_proba_xgb, test_df["amount_clp"].to_numpy(), threshold)

    report = {
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "fraud_rate_overall": float(features["is_fraud"].mean()),
        "decision_threshold": threshold,
        "ensemble": metrics_ensemble,
        "catboost_only": metrics_catboost,
        "xgboost_only": metrics_xgboost,
        "feature_columns": FULL_FEATURE_COLUMNS,
    }

    print(json.dumps(metrics_ensemble, indent=2))

    print("Saving artifacts...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    models.catboost.save_model(str(MODELS_DIR / "catboost_fraud.cbm"))
    models.xgboost.save_model(str(MODELS_DIR / "xgboost_fraud.json"))
    torch.save(ae_artifacts.model.state_dict(), MODELS_DIR / "autoencoder.pt")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    metadata = {
        "feature_columns": NUMERIC_FEATURE_COLUMNS,
        "full_feature_columns": FULL_FEATURE_COLUMNS,
        "autoencoder_n_features": len(NUMERIC_FEATURE_COLUMNS),
        "autoencoder_latent_dim": 4,
        "decision_threshold": threshold,
        "seed": SEED,
    }
    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(REPORTS_DIR / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    importance = dict(zip(FULL_FEATURE_COLUMNS, models.catboost.get_feature_importance().tolist()))
    _save_plots(y_test, test_proba_ensemble, threshold, importance)

    print("[8/9] Training logistic regression baseline (approach 1/3, interpretable)...")
    logistic_model, logistic_scaler = train_logistic_baseline(
        X_train_res, y_train_res, sample_weight_train=sample_weight_res, seed=SEED
    )
    logistic_test_proba = logistic_predict_proba(logistic_model, logistic_scaler, X_test)
    logistic_threshold, _ = find_optimal_threshold(y_val, logistic_predict_proba(logistic_model, logistic_scaler, X_val), val_df["amount_clp"].to_numpy())
    metrics_logistic = evaluate(y_test, logistic_test_proba, test_df["amount_clp"].to_numpy(), logistic_threshold)
    logistic_coefs = coefficient_report(logistic_model, FULL_FEATURE_COLUMNS)
    print(f"  logistic baseline: recall={metrics_logistic['recall']:.3f}, roc_auc={metrics_logistic['roc_auc']:.5f}")

    print("[9/9] Training PyTorch MLP with Focal Loss (approach 3/3, ReLU/GELU/Swish comparison)...")
    X_train_dl = X_train_res.to_numpy(dtype=np.float32)
    X_val_dl = X_val.to_numpy(dtype=np.float32)
    X_test_dl = X_test.to_numpy(dtype=np.float32)
    y_train_dl = y_train_res.astype(np.float32)

    mlp_models, mlp_histories = compare_activations(
        X_train_dl, y_train_dl, X_val_dl, y_val.astype(np.float32), n_epochs=12, seed=SEED
    )
    mlp_metrics = {}
    for activation, model in mlp_models.items():
        val_proba = mlp_predict_proba(model, X_val_dl)
        act_threshold, _ = find_optimal_threshold(y_val, val_proba, val_df["amount_clp"].to_numpy())
        test_proba = mlp_predict_proba(model, X_test_dl)
        mlp_metrics[activation] = evaluate(y_test, test_proba, test_df["amount_clp"].to_numpy(), act_threshold)
        print(f"  mlp[{activation}]: recall={mlp_metrics[activation]['recall']:.3f}, roc_auc={mlp_metrics[activation]['roc_auc']:.5f}")

    best_activation = max(mlp_metrics, key=lambda a: mlp_metrics[a]["roc_auc"])
    best_mlp_model = mlp_models[best_activation]
    best_mlp_proba = mlp_predict_proba(best_mlp_model, X_test_dl)
    torch.save(best_mlp_model.state_dict(), MODELS_DIR / "mlp_focal_best.pt")

    _save_dl_loss_curves(mlp_histories)
    _save_dl_loss_curves_animated(mlp_histories)

    comparison_rows = [
        {"approach": "baseline_interpretable", "model_name": "logistic_regression", **metrics_logistic},
        {"approach": "tree_ensemble", "model_name": "catboost", **metrics_catboost},
        {"approach": "tree_ensemble", "model_name": "xgboost", **metrics_xgboost},
        {"approach": "tree_ensemble", "model_name": "ensemble_avg", **metrics_ensemble},
    ]
    for activation, m in mlp_metrics.items():
        comparison_rows.append({"approach": "deep_learning", "model_name": f"mlp_{activation}_focal_loss", **m})
    _save_model_comparison_plot(comparison_rows)

    print("Persisting comparative metrics to DuckDB...")
    con = metrics_store.connect()
    run_ts = metrics_store.now()
    metrics_store.persist_metrics(con, "baseline_interpretable", "logistic_regression", metrics_logistic, extra={"top_coefficients": dict(list(logistic_coefs.items())[:5])}, run_ts=run_ts)
    metrics_store.persist_metrics(con, "tree_ensemble", "catboost", metrics_catboost, run_ts=run_ts)
    metrics_store.persist_metrics(con, "tree_ensemble", "xgboost", metrics_xgboost, run_ts=run_ts)
    metrics_store.persist_metrics(con, "tree_ensemble", "ensemble_avg", metrics_ensemble, run_ts=run_ts)
    for activation, m in mlp_metrics.items():
        metrics_store.persist_metrics(con, "deep_learning", f"mlp_{activation}_focal_loss", m, extra={"loss_function": "focal_loss", "activation": activation}, run_ts=run_ts)
    metrics_store.persist_predictions(con, "deep_learning_best", y_test, best_mlp_proba, run_ts=run_ts)
    comparison_df = metrics_store.latest_comparison(con)
    con.close()

    report["logistic_baseline"] = metrics_logistic
    report["logistic_top_coefficients"] = dict(list(logistic_coefs.items())[:10])
    report["mlp_focal_loss"] = mlp_metrics
    report["mlp_best_activation"] = best_activation
    with open(REPORTS_DIR / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(REPORTS_DIR / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump({"rows": comparison_rows, "best_by_roc_auc": comparison_df.iloc[0].to_dict() if len(comparison_df) else None}, f, indent=2)

    print(f"Artifacts saved under {MODELS_DIR}, {PLOTS_DIR}, {REPORTS_DIR}")


if __name__ == "__main__":
    main()
