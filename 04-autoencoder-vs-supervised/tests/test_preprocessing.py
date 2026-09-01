import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, dataset_summary, make_splits, normal_only


def _toy_dataframe(n=2000, fraud_rate=0.02, seed=0):
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    data = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    data["Amount"] = np.abs(rng.normal(loc=50, scale=20, size=n))
    data[TARGET_COLUMN] = np.array([1] * n_fraud + [0] * (n - n_fraud))
    return pd.DataFrame(data)


def test_load_raw_strips_openml_quoted_labels(tmp_path):
    from src.data.preprocessing import load_raw

    csv_path = tmp_path / "toy_creditcard.csv"
    df = _toy_dataframe(n=10, fraud_rate=0.2)
    df_quoted = df.copy()
    df_quoted[TARGET_COLUMN] = df_quoted[TARGET_COLUMN].apply(lambda v: f"'{v}'")
    df_quoted.to_csv(csv_path, index=False)

    loaded = load_raw(csv_path)
    assert loaded[TARGET_COLUMN].dtype.kind in "iu"
    assert set(loaded[TARGET_COLUMN].unique()) <= {0, 1}
    assert loaded[TARGET_COLUMN].sum() == df[TARGET_COLUMN].sum()


def test_make_splits_preserves_fraud_ratio():
    df = _toy_dataframe(n=5000, fraud_rate=0.02)
    splits = make_splits(df)

    ratio_total = df[TARGET_COLUMN].mean()
    for y in [splits.y_train, splits.y_val, splits.y_test]:
        assert abs(y.mean() - ratio_total) < 0.01


def test_make_splits_sizes_sum_to_total():
    df = _toy_dataframe(n=1000, fraud_rate=0.02)
    splits = make_splits(df)
    total = len(splits.X_train) + len(splits.X_val) + len(splits.X_test)
    assert total == len(df)


def test_make_splits_scales_time_and_amount_only():
    df = _toy_dataframe(n=1000, fraud_rate=0.02)
    splits = make_splits(df)
    # V1 no deberia estar escalado (se deja tal cual, ya viene como PCA)
    pd.testing.assert_series_equal(
        splits.X_train["V1"].reset_index(drop=True),
        df.loc[splits.X_train.index, "V1"].reset_index(drop=True),
        check_names=False,
    )


def test_normal_only_excludes_fraud():
    df = _toy_dataframe(n=1000, fraud_rate=0.05)
    splits = make_splits(df)
    subset = normal_only(splits.X_train, splits.y_train)
    assert subset.index.isin(splits.y_train[splits.y_train == 0].index).all()
    assert len(subset) == (splits.y_train == 0).sum()


def test_dataset_summary_matches_manual_counts():
    df = _toy_dataframe(n=500, fraud_rate=0.1)
    resumen = dataset_summary(df)
    assert resumen["n_transacciones"] == 500
    assert resumen["n_fraude"] == 50
    assert resumen["n_normal"] == 450
    assert resumen["proporcion_fraude"] == pytest.approx(0.1)
