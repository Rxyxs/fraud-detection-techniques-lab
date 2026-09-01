"""Aplica la optimizacion de umbral sensible a costos
(`cost_sensitive_threshold.py`) sobre los scores de test de cada modelo del
motor (autoencoder, VAE, Deep SVDD, XGBoost supervisado, hibrido), usando
el monto real de cada transaccion (`Amount`, no escalado) como el costo de
cada falso negativo.

Uso:
    python -m src.evaluation.optimize_thresholds
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import load_raw, make_splits
from src.evaluation.cost_sensitive_threshold import (
    DEFAULT_COST_FALSE_POSITIVE,
    cost_at_naive_flag_all,
    optimize_cost_sensitive_threshold,
)

REPORTS_DIR = ROOT / "outputs" / "reports"

SCORE_FILES = {
    "autoencoder_no_supervisado": ("autoencoder_test_scores.parquet", "reconstruction_error"),
    "vae_no_supervisado": ("vae_test_scores.parquet", "vae_score"),
    "deep_svdd_no_supervisado": ("deep_svdd_test_scores.parquet", "deep_svdd_score"),
    "xgboost_supervisado": ("supervised_test_scores.parquet", "xgb_score"),
    "hibrido_xgboost_mas_autoencoder": ("hybrid_test_scores.parquet", "hybrid_score"),
}


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando dataset real y generando particiones (mismo split que todos los modelos)...")
    df = load_raw()
    splits = make_splits(df)
    amounts = df.loc[splits.X_test.index, "Amount"].to_numpy()

    naive_cost = cost_at_naive_flag_all(splits.y_test.to_numpy(), amounts)
    print(f"Costo sin modelo (dejar pasar todo el fraude): ${naive_cost:,.2f}\n")

    resultados = {
        "cost_false_positive_usd": DEFAULT_COST_FALSE_POSITIVE,
        "naive_no_model_cost_usd": naive_cost,
        "n_test": int(len(amounts)),
        "modelos": [],
    }

    for nombre, (archivo, score_col) in SCORE_FILES.items():
        path = REPORTS_DIR / archivo
        if not path.exists():
            print(f"  omitiendo {nombre}: no existe {archivo} (correr el entrenamiento correspondiente primero)")
            continue

        scores_df = pd.read_parquet(path)
        y_true = scores_df["y_true"].to_numpy()
        y_score = scores_df[score_col].to_numpy()
        assert len(y_true) == len(amounts), f"{nombre}: tamano de test no coincide con el split de referencia"

        best, sweep = optimize_cost_sensitive_threshold(y_true, y_score, amounts)
        sweep.to_csv(REPORTS_DIR / f"{nombre}_cost_sweep.csv", index=False)

        reduccion_pct = (1 - best["total_cost_usd"] / naive_cost) * 100
        resultados["modelos"].append({"nombre": nombre, **best, "cost_reduction_vs_naive_pct": reduccion_pct})

        print(f"=== {nombre} ===")
        print(f"  Umbral optimo: {best['threshold']:.6f}  ({best['n_flagged']} alertas, "
              f"{best['tp']} TP / {best['fp']} FP / {best['fn']} FN)")
        print(f"  Costo total: ${best['total_cost_usd']:,.2f}  "
              f"(FP: ${best['cost_false_positives_usd']:,.2f} + FN: ${best['cost_false_negatives_usd']:,.2f})")
        print(f"  Reduccion vs. no usar modelo: {reduccion_pct:.1f}%\n")

    with open(REPORTS_DIR / "cost_sensitive_thresholds.json", "w") as f:
        json.dump(resultados, f, indent=2)

    print(f"Reporte escrito en {REPORTS_DIR / 'cost_sensitive_thresholds.json'}")
    return resultados


if __name__ == "__main__":
    main()
