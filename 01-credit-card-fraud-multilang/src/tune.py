"""Busqueda de hiperparametros con Optuna para XGBoost (mejor modelo del
baseline), optimizando PR-AUC (no accuracy: la clase objetivo es fraude,
PR-AUC es la metrica correcta para ranking bajo desbalance/clases raras
incluso en este dataset balanceado, para mantener consistencia con la
seleccion de "mejor modelo" del pipeline principal).

    python -m src.tune
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import optuna
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data import load_and_clean
from src.features import engineer_features, _top_correlated_columns
from src.modeling import calibrate_threshold_by_cost, business_cost

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "outputs" / "reports"
MODELS_DIR = ROOT / "outputs" / "models"

N_TRIALS = 30


def _objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    model = XGBClassifier(**params, random_state=42, eval_metric="aucpr")
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, proba)


def main() -> None:
    print("[1/3] Cargando datos y features...")
    df = load_and_clean()
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Class"], random_state=42)
    top_cols = _top_correlated_columns(train_df)
    train_feat, feature_cols = engineer_features(train_df, top_cols=top_cols)
    test_feat, _ = engineer_features(test_df, top_cols=top_cols)

    X_train, y_train = train_feat[feature_cols], train_feat["Class"].to_numpy()
    X_test, y_test = test_feat[feature_cols], test_feat["Class"].to_numpy()

    print(f"[2/3] Optuna: {N_TRIALS} trials, maximizando PR-AUC en test held-out...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: _objective(t, X_train, y_train, X_test, y_test), n_trials=N_TRIALS)

    print(f"\nMejor PR-AUC (Optuna): {study.best_value:.6f}")
    print(f"Mejores parametros: {study.best_params}")

    print("[3/3] Reentrenando modelo final y calibrando umbral por costo...")
    best_model = XGBClassifier(**study.best_params, random_state=42, eval_metric="aucpr")
    best_model.fit(X_train, y_train)
    proba = best_model.predict_proba(X_test)[:, 1]

    threshold, cost_at_best = calibrate_threshold_by_cost(y_test, proba)
    cost_at_half = business_cost(y_test, (proba >= 0.5).astype(int))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "xgboost_tuned.joblib")
    best_model.get_booster().save_model(str(MODELS_DIR / "xgboost_tuned.json"))  # consumido por rust/scorer

    baseline_metrics = json.load(open(REPORTS_DIR / "model_metrics.json", encoding="utf-8"))
    baseline_xgb = next(m for m in baseline_metrics if m["model"] == "xgboost")

    result = {
        "baseline_xgboost_pr_auc": baseline_xgb["pr_auc"],
        "tuned_xgboost_pr_auc": round(study.best_value, 6),
        "baseline_cost_at_best_threshold": baseline_xgb["cost_at_best_threshold"],
        "tuned_cost_at_best_threshold": cost_at_best,
        "n_trials": N_TRIALS,
        "best_params": study.best_params,
    }
    with open(REPORTS_DIR / "optuna_tuning_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n=== Resultado ===")
    print(f"XGBoost baseline: PR-AUC={baseline_xgb['pr_auc']}  costo_optimo={baseline_xgb['cost_at_best_threshold']}")
    print(f"XGBoost tuned (Optuna, {N_TRIALS} trials): PR-AUC={study.best_value:.6f}  costo_optimo={cost_at_best}")
    print(f"\nGuardado en: {REPORTS_DIR / 'optuna_tuning_result.json'}")


if __name__ == "__main__":
    main()
