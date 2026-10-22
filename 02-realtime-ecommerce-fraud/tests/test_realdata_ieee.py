"""Schema mapping for the IEEE-CIS comparison (no download needed)."""
import pandas as pd

from src.realdata_ieee import CLP_PER_USD, REFERENCE_DATE, native_features, to_pipeline_schema


def _raw():
    return pd.DataFrame({
        "TransactionID": [1, 2], "TransactionDT": [86400, 90000], "TransactionAmt": [10.0, 2.5],
        "card1": [111, 111], "addr1": [325.0, None], "ProductCD": ["W", "C"],
        "P_emaildomain": ["gmail.com", None], "isFraud": [0, 1],
    })


def test_ieee_rows_map_to_the_pipeline_schema():
    out = to_pipeline_schema(_raw())
    assert list(out["customer_id"]) == ["111_325", "111_-1"]
    assert out["timestamp"].iloc[0] == REFERENCE_DATE + pd.Timedelta(days=1)
    assert out["amount_clp"].iloc[1] == 2.5 * CLP_PER_USD
    assert (out[["latitude", "longitude"]] == 0).all().all()
    assert list(out["is_fraud"]) == [0, 1]


def test_native_features_drop_target_and_encode_categories():
    X = native_features(_raw())
    assert "isFraud" not in X.columns and "TransactionDT" not in X.columns
    assert X["ProductCD"].tolist() == [1.0, 0.0]
    assert X["P_emaildomain"].tolist() == [0.0, -1.0]
