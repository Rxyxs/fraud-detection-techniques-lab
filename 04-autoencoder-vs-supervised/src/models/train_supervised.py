"""Entrena el modelo supervisado de comparacion (XGBoost) sobre el mismo
split de datos, aprovechando que este dataset SI tiene fraude confirmado
(a diferencia del proyecto de deteccion de LA basado en grafos, donde la
etiqueta nunca existe en un despliegue real). El desbalance extremo
(~0.17% fraude) se maneja con `scale_pos_weight`, no con oversampling
sintetico.

Uso:
    python -m src.models.train_supervised
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import make_splits, load_raw

OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"

RANDOM_STATE = 42


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando dataset real y generando particiones (mismo split que el autoencoder)...")
    df = load_raw()
    splits = make_splits(df)

    n_neg = int((splits.y_train == 0).sum())
    n_pos = int((splits.y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos
    print(f"Train: {n_neg:,} normales / {n_pos:,} fraude -> scale_pos_weight={scale_pos_weight:.1f}")

    modelo = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    modelo.fit(splits.X_train, splits.y_train, eval_set=[(splits.X_val, splits.y_val)], verbose=False)

    y_score_test = modelo.predict_proba(splits.X_test)[:, 1]
    roc_auc = roc_auc_score(splits.y_test, y_score_test)
    pr_auc = average_precision_score(splits.y_test, y_score_test)
    print(f"\nTest ROC-AUC: {roc_auc:.4f} | Test PR-AUC (average precision): {pr_auc:.4f}")

    joblib.dump(modelo, MODELS_DIR / "xgboost_supervisado.joblib")

    resultados_test = pd.DataFrame({
        "xgb_score": y_score_test,
        "y_true": splits.y_test.to_numpy(),
    })
    resultados_test.to_parquet(REPORTS_DIR / "supervised_test_scores.parquet")

    print("Calculando importancia de features (SHAP) sobre una muestra del test set...")
    muestra = splits.X_test.sample(n=min(2000, len(splits.X_test)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(muestra)

    importancia = pd.DataFrame({
        "feature": muestra.columns,
        "importancia_media_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("importancia_media_abs_shap", ascending=False)
    importancia.to_csv(REPORTS_DIR / "shap_feature_importance.csv", index=False)
    print(importancia.head(10).to_string(index=False))

    np.save(REPORTS_DIR / "shap_values_sample.npy", shap_values)
    muestra.to_parquet(REPORTS_DIR / "shap_sample_features.parquet")

    print(f"\nArtefactos escritos en {MODELS_DIR} y {REPORTS_DIR}")
    return modelo, resultados_test, importancia


if __name__ == "__main__":
    main()
