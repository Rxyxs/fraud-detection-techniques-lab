"""Diagnostico: como cambia la deteccion del autoencoder segun cuanto entrena.

Entrena el mismo autoencoder (``deep_baseline._train_autoencoder``) con un
numero fijo de epocas -- sin early stopping -- y mide ROC-AUC y average
precision contra ``es_ilicito``. Incluye la distancia cuadratica media a la
media de las features estandarizadas, que es lo que un autoencoder casi sin
entrenar termina calculando.

Es un diagnostico, no un criterio de seleccion: elegir el numero de epocas
mirando ``es_ilicito`` seria usar las etiquetas para entrenar un detector
que se presenta como no supervisado.

    python -m src.anomaly.autoencoder_epoch_sweep
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from src.anomaly.deep_baseline import ACTIVATIONS, _feature_matrix, _train_autoencoder
from src.anomaly.run_model_comparison import (
    DATA_DIR,
    END_DATE,
    UMBRAL_ESTRUCTURACION_CLP,
    build_feature_table,
    build_transaction_features,
    build_weighted_digraph,
    ensure_all_accounts_present,
)

EPOCHS = (60, 100, 200, 500, 1000, 2000)
OUTPUT = DATA_DIR.parents[1] / "outputs" / "autoencoder_epoch_sweep.json"


def distance_to_mean(X: np.ndarray) -> np.ndarray:
    sigma = X.std(axis=0)
    Z = (X - X.mean(axis=0)) / np.where(sigma < 1e-6, 1e-6, sigma)
    return (Z ** 2).mean(axis=1)


def run() -> dict:
    accounts = pl.read_parquet(DATA_DIR / "accounts.parquet")
    transfers = pl.read_parquet(DATA_DIR / "transfers.parquet")
    graph = ensure_all_accounts_present(build_weighted_digraph(transfers), accounts["account_id"].to_list())
    account_features = build_feature_table(graph, transfers, accounts, UMBRAL_ESTRUCTURACION_CLP, END_DATE)
    tx = build_transaction_features(transfers, account_features, UMBRAL_ESTRUCTURACION_CLP)
    X, y = _feature_matrix(tx), tx["es_ilicito"].to_numpy()

    d = distance_to_mean(X)
    out = {"distance_to_mean": {"roc_auc": roc_auc_score(y, d), "average_precision": average_precision_score(y, d)}}
    for activation in ACTIVATIONS:
        out[activation] = {}
        for epochs in EPOCHS:
            score, history = _train_autoencoder(X, activation, epochs=epochs, min_epochs=epochs,
                                                patience=10**9, return_history=True)
            out[activation][epochs] = {
                "roc_auc": roc_auc_score(y, score),
                "average_precision": average_precision_score(y, score),
                "val_mse": min(history["val_loss"]),
                "corr_with_distance_to_mean": float(np.corrcoef(score, d)[0, 1]),
            }
            print(f"{activation:6s} {epochs:5d} epocas  ROC-AUC={out[activation][epochs]['roc_auc']:.3f}  "
                  f"AP={out[activation][epochs]['average_precision']:.3f}  "
                  f"MSE val={out[activation][epochs]['val_mse']:.2f}  "
                  f"corr dist={out[activation][epochs]['corr_with_distance_to_mean']:.3f}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    run()
