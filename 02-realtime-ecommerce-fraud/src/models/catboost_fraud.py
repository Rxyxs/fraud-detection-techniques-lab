"""Cost-sensitive supervised classifiers (CatBoost + XGBoost) for the final
fraud decision.

Cost-sensitivity is implemented two ways, deliberately kept separate:

1. **Training-time**: per-sample weights (``compute_sample_weights``) that
   scale each fraudulent row by its own CLP amount relative to the average
   fraud amount, on top of the usual class-imbalance ratio -- a missed
   CLP 2,000,000 fraud is a worse training error than a missed CLP 5,000 one.
2. **Decision-time**: the classification threshold is chosen by
   ``find_optimal_threshold`` to minimize an explicit business cost function
   (money lost to false negatives + review cost of false positives), rather
   than defaulting to the conventional 0.5 cutoff, which is meaningless under
   ~99% class imbalance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Estimated operational cost of a manual fraud review / customer contact
# triggered by a false alarm. Approximate CLP figure for a contact-center
# interaction; used only to make the cost trade-off concrete and auditable.
DEFAULT_REVIEW_COST_CLP = 3_000.0


def compute_sample_weights(y: np.ndarray, amount_clp: np.ndarray) -> np.ndarray:
    """Cost-sensitive training weights.

    Fraud rows are weighted by the class-imbalance ratio (n_neg / n_pos)
    times their own amount relative to the average fraud amount, so the
    model is pushed harder to catch high-value fraud than low-value fraud.
    Legit rows all get weight 1.
    """
    y = np.asarray(y)
    amount_clp = np.asarray(amount_clp, dtype=float)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    class_ratio = n_neg / max(n_pos, 1)

    fraud_amounts = amount_clp[y == 1]
    avg_fraud_amount = fraud_amounts.mean() if len(fraud_amounts) else 1.0

    weights = np.ones(len(y), dtype=float)
    weights[y == 1] = class_ratio * (amount_clp[y == 1] / avg_fraud_amount)
    return weights


def resample_train_split(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42):
    """SMOTE (oversample the minority class) combined with Tomek Links
    (remove ambiguous majority-class points sitting on the decision
    boundary) -- applied to the TRAIN split only, never validation/test.
    """
    smt = SMOTETomek(random_state=seed)
    X_res, y_res = smt.fit_resample(X_train, y_train)
    return X_res, y_res


@dataclass
class TrainedModels:
    catboost: CatBoostClassifier
    xgboost: XGBClassifier


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight_train: np.ndarray | None = None,
    seed: int = 42,
) -> TrainedModels:
    cat_model = CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
    )
    cat_model.fit(X_train, y_train, sample_weight=sample_weight_train)

    xgb_model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, sample_weight=sample_weight_train)

    return TrainedModels(catboost=cat_model, xgboost=xgb_model)


def business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amount_clp: np.ndarray,
    review_cost: float = DEFAULT_REVIEW_COST_CLP,
) -> tuple[float, float, float]:
    """Total business cost = CLP lost to missed fraud (false negatives) +
    fixed review cost per false alarm (false positives)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    amount_clp = np.asarray(amount_clp, dtype=float)

    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_cost = float(amount_clp[fn_mask].sum())
    fp_cost = float(review_cost * fp_mask.sum())
    return fn_cost + fp_cost, fn_cost, fp_cost


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    amount_clp: np.ndarray,
    review_cost: float = DEFAULT_REVIEW_COST_CLP,
    n_steps: int = 200,
) -> tuple[float, float]:
    """Scans candidate thresholds and returns the one minimizing
    ``business_cost`` -- the cost-sensitive analogue of picking 0.5."""
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_threshold, best_cost = 0.5, np.inf
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        total_cost, _, _ = business_cost(y_true, y_pred, amount_clp, review_cost)
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = float(t)
    return best_threshold, best_cost


def evaluate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    amount_clp: np.ndarray,
    threshold: float,
    review_cost: float = DEFAULT_REVIEW_COST_CLP,
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_cost, fn_cost, fp_cost = business_cost(y_true, y_pred, amount_clp, review_cost)
    no_model_cost = float(np.asarray(amount_clp)[y_true == 1].sum())

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "total_cost_clp": total_cost,
        "false_negative_cost_clp": fn_cost,
        "false_positive_cost_clp": fp_cost,
        "no_model_baseline_cost_clp": no_model_cost,
        "cost_savings_vs_no_model_clp": no_model_cost - total_cost,
    }
