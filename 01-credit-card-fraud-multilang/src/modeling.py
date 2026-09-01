"""Iteracion de modelos: LogisticRegression+SMOTE (baseline) -> CatBoost ->
XGBoost -> calibracion de umbral de decision por matriz de costo de
negocio (falso positivo vs. perdida por fraude no detectado), no el 0.5
por defecto."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42

# Matriz de costo de negocio (valores ilustrativos pero realistas para
# tarjetas de credito de consumo): dejar pasar un fraude cuesta ~10x mas
# que investigar una transaccion legitima marcada por error.
COST_FALSE_NEGATIVE = 100.0  # fraude no detectado
COST_FALSE_POSITIVE = 10.0   # investigacion de transaccion legitima


def train_logreg_smote(X_train, y_train) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_scaled, y_train)
    model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    model.fit(X_res, y_res)
    return model, scaler


def train_catboost(X_train, y_train) -> CatBoostClassifier:
    model = CatBoostClassifier(
        iterations=400, depth=6, learning_rate=0.05, random_seed=RANDOM_STATE,
        verbose=False, allow_writing_files=False,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=RANDOM_STATE, eval_metric="aucpr",
    )
    model.fit(X_train, y_train)
    return model


def business_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE


def calibrate_threshold_by_cost(y_true: np.ndarray, y_proba: np.ndarray, n_thresholds: int = 200) -> tuple[float, float]:
    """Barre umbrales y devuelve (umbral_optimo, costo_minimo) segun la
    matriz de costo de negocio -- no el umbral 0.5 por defecto."""
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    best_threshold, best_cost = 0.5, float("inf")
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        cost = business_cost(y_true, pred)
        if cost < best_cost:
            best_cost, best_threshold = cost, t
    return best_threshold, best_cost


def evaluate_model(name: str, y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    threshold, cost_at_best = calibrate_threshold_by_cost(y_true, y_proba)
    pred_default = (y_proba >= 0.5).astype(int)
    pred_calibrated = (y_proba >= threshold).astype(int)
    return {
        "model": name,
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "cost_at_threshold_0.5": business_cost(y_true, pred_default),
        "best_threshold": round(float(threshold), 4),
        "cost_at_best_threshold": cost_at_best,
        "cost_reduction_pct": round(
            100 * (1 - cost_at_best / max(business_cost(y_true, pred_default), 1)), 2
        ),
    }
