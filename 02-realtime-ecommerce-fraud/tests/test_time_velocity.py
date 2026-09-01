import numpy as np
import pandas as pd

from src.features.time_velocity import add_time_velocity_features


def _make_df():
    # Customer 1: a regular monthly cadence, then a burst of 3 rapid transactions.
    return pd.DataFrame({
        "customer_id": [1, 1, 1, 1, 1, 2],
        "timestamp": pd.to_datetime([
            "2026-01-01 10:00:00",
            "2026-01-15 10:00:00",  # 14 days later (regular cadence)
            "2026-01-29 10:00:00",  # 14 days later (regular cadence)
            "2026-01-29 10:02:00",  # 2 minutes later -> burst
            "2026-01-29 10:04:00",  # 2 minutes later -> burst
            "2026-01-01 09:00:00",
        ]),
        "amount_clp": [10000, 11000, 9000, 500, 500, 5000],
    })


def test_first_transaction_has_no_prior_gap():
    df = add_time_velocity_features(_make_df())
    first_row = df[df["customer_id"] == 1].iloc[0]
    assert np.isinf(first_row["seconds_since_prev"])
    assert first_row["velocity_ratio"] == 1.0  # fallback for "no history yet"


def test_velocity_ratio_flags_burst():
    df = add_time_velocity_features(_make_df())
    cust1 = df[df["customer_id"] == 1].reset_index(drop=True)
    # The 4th and 5th transactions arrive far faster than this customer's own
    # ~14-day historical cadence -> velocity_ratio should be << 1.
    assert cust1.loc[3, "velocity_ratio"] < 0.01
    assert cust1.loc[4, "velocity_ratio"] < 0.01


def test_txn_count_last_1h_counts_burst_not_regular_cadence():
    df = add_time_velocity_features(_make_df())
    cust1 = df[df["customer_id"] == 1].reset_index(drop=True)
    # Row 2 (2026-01-15) has no other transaction from this customer in the
    # trailing hour.
    assert cust1.loc[1, "txn_count_last_1h"] == 0
    # Row 4 (10:02) has exactly one other transaction (row 3, 10:00) in the
    # trailing hour.
    assert cust1.loc[3, "txn_count_last_1h"] == 1
    # Row 5 (10:04) has two prior transactions (10:00 and 10:02) in the
    # trailing hour.
    assert cust1.loc[4, "txn_count_last_1h"] == 2


def test_independent_customers_dont_leak():
    df = add_time_velocity_features(_make_df())
    cust2 = df[df["customer_id"] == 2].reset_index(drop=True)
    assert np.isinf(cust2.loc[0, "seconds_since_prev"])
    assert cust2.loc[0, "txn_count_last_1h"] == 0
