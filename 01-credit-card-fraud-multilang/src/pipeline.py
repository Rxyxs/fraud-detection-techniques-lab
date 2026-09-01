"""Pipeline end-to-end: limpieza -> feature engineering -> iteracion de
modelos (LogReg+SMOTE -> CatBoost -> XGBoost -> MLP+Focal Loss) ->
validacion adversaria -> calibracion de umbral por costo -> graficos ->
ONNX + SQLite.

    python -m src.pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.data import load_and_clean
from src.features import engineer_features, _top_correlated_columns
from src.adversarial_validation import adversarial_validation_auc
from src.deep import compare_activations, predict_proba, train_mlp
from src.modeling import evaluate_model, train_catboost, train_logreg_smote, train_xgboost
from src.plots import plot_loss_curves, plot_loss_curves_animated, plot_model_comparison_with_mlp
from src.storage import export_results

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"
REPORTS_DIR = ROOT / "outputs" / "reports"


def main() -> None:
    print("[1/6] Cargando y limpiando dataset real (568k transacciones, 2023)...")
    df = load_and_clean()

    print("[2/6] Split train/test (80/20, estratificado)...")
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Class"], random_state=42)

    print("[3/6] Feature engineering (interacciones sobre componentes PCA top-correlacionados + Amount)...")
    top_cols = _top_correlated_columns(train_df)
    train_feat, feature_cols = engineer_features(train_df, top_cols=top_cols)
    test_feat, _ = engineer_features(test_df, top_cols=top_cols)

    X_train, y_train = train_feat[feature_cols], train_feat["Class"].to_numpy()
    X_test, y_test = test_feat[feature_cols], test_feat["Class"].to_numpy()

    print("[4/6] Validacion adversaria (train vs. test son intercambiables?)...")
    adv_auc = adversarial_validation_auc(X_train, X_test)
    print(f"  AUC adversario = {adv_auc:.4f} (0.5 = indistinguibles, splits sanos)")

    print("[5/6] Iteracion de modelos: LogReg+SMOTE -> CatBoost -> XGBoost...")
    results = []

    logreg, scaler = train_logreg_smote(X_train, y_train)
    logreg_proba = logreg.predict_proba(scaler.transform(X_test))[:, 1]
    results.append(evaluate_model("logreg_smote", y_test, logreg_proba))

    catboost = train_catboost(X_train, y_train)
    catboost_proba = catboost.predict_proba(X_test)[:, 1]
    results.append(evaluate_model("catboost", y_test, catboost_proba))

    xgb = train_xgboost(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    results.append(evaluate_model("xgboost", y_test, xgb_proba))

    print("[5b/6] Deep learning: MLP + Focal Loss, comparando ReLU vs. GELU vs. Swish...")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    activation_models, activation_history = compare_activations(
        X_train_scaled, y_train, X_test_scaled, y_test, n_epochs=15,
    )
    best_activation = (
        activation_history[activation_history["epoch"] == activation_history["epoch"].max()]
        .set_index("activation")["val_loss"].idxmin()
    )
    print(f"  Mejor activacion por val_loss final: {best_activation}")
    mlp_proba = predict_proba(activation_models[best_activation], X_test_scaled)
    results.append(evaluate_model("mlp_focal_loss", y_test, mlp_proba))

    results_df = pd.DataFrame(results)
    print("\n=== Metricas por modelo (test set, 20% held-out) ===")
    print(results_df.to_string(index=False))

    best_name = results_df.loc[results_df["pr_auc"].idxmax(), "model"]
    print(f"\nMejor modelo por PR-AUC: {best_name}")

    print("[6/6] Graficos, exportando CatBoost a ONNX (soporte nativo) + SQLite...")
    plot_loss_curves(activation_history)
    plot_loss_curves_animated(activation_history)
    plot_model_comparison_with_mlp(results_df)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    catboost.save_model(str(MODELS_DIR / "catboost_model.onnx"), format="onnx")
    joblib.dump({"logreg": logreg, "scaler": scaler, "catboost": catboost, "xgb": xgb}, MODELS_DIR / "all_models.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_columns.joblib")
    joblib.dump(top_cols, MODELS_DIR / "top_correlated_columns.joblib")
    torch.save(activation_models[best_activation].state_dict(), MODELS_DIR / "mlp_focal_loss.pt")

    results_df.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
    activation_history.to_csv(REPORTS_DIR / "mlp_activation_history.csv", index=False)
    summary = {
        "adversarial_validation_auc": adv_auc,
        "best_model": best_name,
        "best_mlp_activation": best_activation,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "fraud_rate_pct": round(100 * df["Class"].mean(), 3),
    }
    with open(REPORTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    predictions_df = pd.DataFrame({
        "y_true": y_test,
        "proba_catboost": catboost_proba,
        "proba_xgboost": xgb_proba,
        "proba_mlp_focal_loss": mlp_proba,
    })
    export_results(results_df, predictions_df)

    print(f"\nGuardado en: {MODELS_DIR}, {REPORTS_DIR}, outputs/fraud.sqlite")


if __name__ == "__main__":
    main()
