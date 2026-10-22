"""Scoring helper of the Elliptic comparison (no download needed)."""
import numpy as np

from src.realdata_elliptic import score


def test_score_ignores_unknown_labels_and_detects_inverted_rankings():
    labels = np.array([1, 0, 0, 0, -1, -1])
    good = score(labels, np.array([0.9, 0.1, 0.2, 0.3, 5.0, 5.0]))
    inverted = score(labels, np.array([0.0, 0.5, 0.6, 0.7, 9.0, 9.0]))
    assert good["roc_auc"] == 1.0 and good["precision_at_base_rate"] == 1.0
    assert inverted["roc_auc"] == 0.0
