"""Experimento honesto: ¿ayuda agregar el error de reconstruccion del
autoencoder no supervisado como feature extra de un XGBoost supervisado?

Reentrena un segundo XGBoost identico al de `train_supervised.py` pero con
una columna adicional (`ae_reconstruction_error`), calculada aplicando el
autoencoder ya entrenado (solo con normales) sobre TODAS las filas de cada
particion. El resultado puede ir en cualquier direccion — este script no
asume de antemano que el hibrido gane.

Uso:
    python -m src.models.train_hybrid
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import load_raw, make_splits
from src.models.autoencoder import FraudAutoencoder, reconstruction_error
from src.models.train_supervised import RANDOM_STATE

OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"


def _to_tensor(df: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(df.to_numpy(dtype=np.float32))


def main():
    df = load_raw()
    splits = make_splits(df)

    n_features = len(splits.X_train.columns)
    autoencoder = FraudAutoencoder(input_dim=n_features)
    autoencoder.load_state_dict(torch.load(MODELS_DIR / "autoencoder.pt"))

    X_train = splits.X_train.copy()
    X_val = splits.X_val.copy()
    X_test = splits.X_test.copy()
    X_train["ae_reconstruction_error"] = reconstruction_error(autoencoder, _to_tensor(splits.X_train)).numpy()
    X_val["ae_reconstruction_error"] = reconstruction_error(autoencoder, _to_tensor(splits.X_val)).numpy()
    X_test["ae_reconstruction_error"] = reconstruction_error(autoencoder, _to_tensor(splits.X_test)).numpy()

    scale_pos_weight = (splits.y_train == 0).sum() / (splits.y_train == 1).sum()
    modelo_hibrido = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        random_state=RANDOM_STATE, n_jobs=1,
    )
    modelo_hibrido.fit(X_train, splits.y_train, eval_set=[(X_val, splits.y_val)], verbose=False)

    y_score = modelo_hibrido.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(splits.y_test, y_score)
    pr_auc = average_precision_score(splits.y_test, y_score)

    baseline = pd.read_parquet(REPORTS_DIR / "supervised_test_scores.parquet")
    baseline_roc = roc_auc_score(baseline.y_true, baseline.xgb_score)
    baseline_pr = average_precision_score(baseline.y_true, baseline.xgb_score)

    print(f"XGBoost supervisado (baseline):        ROC-AUC={baseline_roc:.4f}  PR-AUC={baseline_pr:.4f}")
    print(f"XGBoost + feature autoencoder (hibrido): ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")
    print(f"Delta PR-AUC: {pr_auc - baseline_pr:+.4f}")

    pd.DataFrame({"hybrid_score": y_score, "y_true": splits.y_test.to_numpy()}).to_parquet(
        REPORTS_DIR / "hybrid_test_scores.parquet"
    )

    import json
    with open(REPORTS_DIR / "hybrid_comparison.json", "w") as f:
        json.dump({
            "baseline_roc_auc": float(baseline_roc), "baseline_pr_auc": float(baseline_pr),
            "hybrid_roc_auc": float(roc_auc), "hybrid_pr_auc": float(pr_auc),
            "delta_pr_auc": float(pr_auc - baseline_pr),
        }, f, indent=2)


if __name__ == "__main__":
    main()
