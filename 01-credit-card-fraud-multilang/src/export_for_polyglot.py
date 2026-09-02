"""Exporta artefactos que consumen los componentes en otros lenguajes
(rust/scorer, julia/cost_sensitivity.jl): probabilidades de referencia del
XGBoost afinado, para verificacion bit-a-bit y para el analisis de
sensibilidad de costo. Requiere haber corrido `python -m src.tune` antes
(necesita xgboost_tuned.joblib).

    python -m src.export_for_polyglot
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import load_and_clean
from src.features import engineer_features, _top_correlated_columns

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"
REPORTS_DIR = ROOT / "outputs" / "reports"

RUST_VERIFICATION_ROWS = 2000


def main() -> None:
    print("[1/3] Cargando datos y el modelo XGBoost afinado...")
    df = load_and_clean()
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Class"], random_state=42)
    top_cols = _top_correlated_columns(train_df)
    test_feat, _ = engineer_features(test_df, top_cols=top_cols)

    model = joblib.load(MODELS_DIR / "xgboost_tuned.joblib")
    feature_cols = joblib.load(MODELS_DIR / "feature_columns.joblib")
    proba_full = model.predict_proba(test_feat[feature_cols])[:, 1]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[2/3] Exportando referencia de verificacion para Rust ({RUST_VERIFICATION_ROWS} filas)...")
    pd.DataFrame({"proba": proba_full[:RUST_VERIFICATION_ROWS]}).to_csv(
        REPORTS_DIR / "rust_verification_reference.csv", index=False
    )
    # Las mismas RUST_VERIFICATION_ROWS filas del test set (V1-V28 + Amount,
    # crudas, sin feature engineering) en el mismo orden que las probabilidades
    # de arriba -- train_test_split(shuffle=True) reordena las filas respecto
    # al CSV original, asi que Rust no puede simplemente leer las primeras N
    # filas de creditcard_2023.csv: necesita las filas de test reales, en
    # este orden, para que la verificacion bit-a-bit compare la misma
    # transaccion en ambos lenguajes.
    v_cols = [f"V{i}" for i in range(1, 29)]
    test_df.iloc[:RUST_VERIFICATION_ROWS][v_cols + ["Amount"]].to_csv(
        REPORTS_DIR / "rust_verification_rows.csv", index=False
    )

    print(f"[3/3] Exportando predicciones completas del test set para Julia ({len(proba_full)} filas)...")
    pd.DataFrame({"proba": proba_full, "y_true": test_df["Class"].to_numpy()}).to_csv(
        REPORTS_DIR / "xgboost_full_test_predictions.csv", index=False
    )

    print(f"\nGuardado en: {REPORTS_DIR}/rust_verification_reference.csv, xgboost_full_test_predictions.csv")


if __name__ == "__main__":
    main()
