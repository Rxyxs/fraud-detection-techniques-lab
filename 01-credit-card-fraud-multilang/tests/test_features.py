import pandas as pd

from src.features import engineer_features


def _toy_df():
    data = {f"V{i}": [0.1 * i, -0.1 * i, 0.2 * i] for i in range(1, 29)}
    data["Amount"] = [10.0, 500.0, 0.0]
    data["Class"] = [0, 1, 0]
    return pd.DataFrame(data)


def test_amount_log_is_monotonic_with_amount():
    df = _toy_df()
    feat, _ = engineer_features(df, top_cols=["V1", "V2"])
    assert feat["amount_log"].iloc[1] > feat["amount_log"].iloc[0] > feat["amount_log"].iloc[2]


def test_v_l2_norm_nonnegative():
    df = _toy_df()
    feat, _ = engineer_features(df, top_cols=["V1", "V2"])
    assert (feat["v_l2_norm"] >= 0).all()


def test_interaction_columns_created_for_top_cols():
    df = _toy_df()
    feat, cols = engineer_features(df, top_cols=["V1", "V2", "V3"])
    assert "V1_x_V2" in cols
    assert "V1_x_V3" in cols
    assert "V2_x_V3" in cols
