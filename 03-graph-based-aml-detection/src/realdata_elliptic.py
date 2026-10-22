"""The unsupervised AML ensemble on a real transaction graph: the Elliptic dataset.

The synthetic Chilean transfer network plants four AML typologies, and the
IForest + COPOD + ECOD ensemble over graph features finds them with ROC-AUC 0.893.
Elliptic (Weber et al. 2019, Kaggle `ellipticco/elliptic-data-set`) is real:
203,769 Bitcoin transactions linked by 234,355 payment flows, over 49 time steps
about two weeks apart. 4,545 are labeled illicit (scams, malware, ransomware,
Ponzi schemes...), 42,019 licit, and the rest unknown.

Same ensemble, same graph metrics (`compute_graph_metrics`, with every edge
weight set to 1 because Elliptic publishes no amounts), three feature sets:

- graph metrics only, the structural view the module is built on;
- Elliptic's 93 local transaction features (anonymized, per transaction);
- both.

Scores are evaluated on labeled transactions only (illicit vs. licit). As a ceiling,
a supervised Random Forest on the same local features is trained on time steps 1-34
and tested on 35-49, the split the dataset's authors used; time step 43 is when a
large dark market shut down, after which published models stop detecting illicit
transactions, so the per-step recall is reported as well.

    python -m src.realdata_elliptic --dir data/raw/elliptic/elliptic_bitcoin_dataset
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
from pyod.models.combination import average
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.utils.utility import standardizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from src.graph.graph_features import compute_graph_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "outputs" / "reports" / "real_elliptic_metrics.json"
GRAPH_COLUMNS = ["grado_entrada", "grado_salida", "pagerank", "coef_agrupamiento",
                 "centralidad_intermediacion", "reciprocidad_local", "tamanio_componente_conexa"]
N_LOCAL = 93  # columns 2..94 of elliptic_txs_features.csv are the local features
TRAIN_LAST_STEP = 34
DARK_MARKET_SHUTDOWN_STEP = 43


def load(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = pd.read_csv(directory / "elliptic_txs_features.csv", header=None)
    feats.columns = ["txId", "time_step"] + [f"local_{i}" for i in range(N_LOCAL)] + [
        f"agg_{i}" for i in range(feats.shape[1] - 2 - N_LOCAL)]
    classes = pd.read_csv(directory / "elliptic_txs_classes.csv")
    feats = feats.merge(classes, on="txId")
    feats["label"] = feats["class"].map({"1": 1, "2": 0, "unknown": -1}).astype(int)
    edges = pd.read_csv(directory / "elliptic_txs_edgelist.csv")
    return feats, edges


def graph_metrics(feats: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    graph = nx.DiGraph()
    graph.add_nodes_from(feats["txId"])
    graph.add_edges_from(zip(edges["txId1"], edges["txId2"]), weight=1.0, monto_total_clp=1.0)
    metrics = compute_graph_metrics(graph).to_pandas().rename(columns={"account_id": "txId"})
    return feats.merge(metrics[["txId", *GRAPH_COLUMNS]], on="txId")


def ensemble_scores(X: np.ndarray, seed: int = 42) -> np.ndarray:
    detectors = [IForest(n_estimators=200, random_state=seed, n_jobs=-1), COPOD(), ECOD()]
    raw = np.column_stack([d.fit(X).decision_scores_ for d in detectors])
    return average(standardizer(raw))


def score(labels: np.ndarray, scores: np.ndarray) -> dict:
    mask = labels >= 0
    y, s = labels[mask], scores[mask]
    top = s >= np.quantile(s, 1 - y.mean())  # flag as many as there are illicit
    return {"roc_auc": round(float(roc_auc_score(y, s)), 4),
            "average_precision": round(float(average_precision_score(y, s)), 4),
            "precision_at_base_rate": round(float(y[top].mean()), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=ROOT / "data" / "raw" / "elliptic" / "elliptic_bitcoin_dataset")
    args = parser.parse_args()
    start = time.perf_counter()

    feats, edges = load(args.dir)
    print(f"Elliptic: {len(feats):,} transactions, {len(edges):,} edges, "
          f"{(feats.label == 1).sum():,} illicit / {(feats.label == 0).sum():,} licit")
    t0 = time.perf_counter()
    feats = graph_metrics(feats, edges)
    print(f"graph metrics in {time.perf_counter() - t0:.0f} s")

    local_cols = [f"local_{i}" for i in range(N_LOCAL)]
    labels = feats["label"].to_numpy()
    unsupervised = {}
    for name, cols in (("graph_only", GRAPH_COLUMNS), ("local_only", local_cols), ("graph_plus_local", GRAPH_COLUMNS + local_cols)):
        unsupervised[name] = score(labels, ensemble_scores(feats[cols].to_numpy(float)))
        print(name, unsupervised[name])

    labeled = feats[feats.label >= 0]
    train = labeled[labeled.time_step <= TRAIN_LAST_STEP]
    test = labeled[labeled.time_step > TRAIN_LAST_STEP]
    rf = RandomForestClassifier(n_estimators=100, max_features=50, n_jobs=-1, random_state=42)
    rf.fit(train[local_cols], train["label"])
    proba = rf.predict_proba(test[local_cols])[:, 1]
    pred = (proba >= 0.5).astype(int)
    per_step = (test.assign(pred=pred).query("label == 1").groupby("time_step")["pred"].mean().round(3))
    supervised = {
        "roc_auc": round(float(roc_auc_score(test["label"], proba)), 4),
        "average_precision": round(float(average_precision_score(test["label"], proba)), 4),
        "precision": round(float(precision_score(test["label"], pred)), 4),
        "recall": round(float(recall_score(test["label"], pred)), 4),
        "f1": round(float(f1_score(test["label"], pred)), 4),
        "recall_before_shutdown": round(float(pred[(test.label == 1).to_numpy() & (test.time_step < DARK_MARKET_SHUTDOWN_STEP).to_numpy()].mean()), 4),
        "recall_from_shutdown": round(float(pred[(test.label == 1).to_numpy() & (test.time_step >= DARK_MARKET_SHUTDOWN_STEP).to_numpy()].mean()), 4),
        "illicit_recall_by_step": {int(k): float(v) for k, v in per_step.items()},
    }
    print("supervised RF", {k: v for k, v in supervised.items() if k != "illicit_recall_by_step"})

    summary = {
        "dataset": "ellipticco/elliptic-data-set",
        "n_transactions": int(len(feats)),
        "n_edges": int(len(edges)),
        "n_illicit": int((labels == 1).sum()),
        "n_licit": int((labels == 0).sum()),
        "illicit_share_of_labeled": round(float((labels == 1).sum() / (labels >= 0).sum()), 4),
        "unsupervised_ensemble": unsupervised,
        "supervised_random_forest_temporal": supervised,
        "elapsed_seconds": round(time.perf_counter() - start, 1),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
