"""Validacion adversaria: entrena un clasificador para distinguir filas de
train vs. test. Si logra separar bien (AUC alto), train y test NO son
intercambiables -- hay una diferencia de distribucion real entre ambos, y
cualquier modelo entrenado en train puede fallar silenciosamente en test.

Nota de alcance honesta: el dataset no tiene timestamp real (es tabular
puro, PCA-anonimizado), por lo que esto no mide "deriva temporal" en
sentido literal como en IEEE-CIS -- mide si el split hold-out aleatorio
introduce alguna asimetria detectable, la version generica y aun asi
genuinamente util de la misma tecnica."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score


def adversarial_validation_auc(X_train: pd.DataFrame, X_test: pd.DataFrame, random_state: int = 42) -> float:
    combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    is_test = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

    model = LGBMClassifier(n_estimators=200, num_leaves=15, random_state=random_state, verbose=-1)
    scores = cross_val_score(model, combined, is_test, cv=3, scoring="roc_auc")
    return float(scores.mean())
