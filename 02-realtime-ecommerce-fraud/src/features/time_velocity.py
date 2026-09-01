"""Temporal / velocity features: transaction burst rate relative to each
customer's own historical cadence -- the primary signature of card-testing
and account-takeover fraud (many transactions in a very short span).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds, per customer (grouped, ordered by timestamp), with no lookahead:

    - ``seconds_since_prev``: gap to the previous transaction (``inf`` for a
      customer's first transaction).
    - ``avg_seconds_between_txn``: the customer's own expanding average gap,
      computed only from transactions strictly before this one.
    - ``velocity_ratio``: ``seconds_since_prev / avg_seconds_between_txn``.
      Values << 1 mean this transaction arrived much faster than the
      customer's normal cadence -- the core "burst" signal.
    - ``txn_count_last_1h`` / ``txn_count_last_24h``: number of *other*
      transactions by the same customer within the trailing window.
    - ``amount_sum_last_1h``: CLP volume by the same customer within the
      trailing 1-hour window (excludes the current transaction's own amount).

    Assumes rows are already sorted by (customer_id, timestamp).
    """
    df = df.copy()
    grp = df.groupby("customer_id", sort=False)

    seconds_since_prev = grp["timestamp"].diff().dt.total_seconds()
    df["seconds_since_prev"] = seconds_since_prev.fillna(np.inf)

    gap_no_inf = df["seconds_since_prev"].replace(np.inf, np.nan)
    avg_gap = gap_no_inf.groupby(df["customer_id"], group_keys=False).apply(
        lambda s: s.shift(1).expanding().mean()
    )
    df["avg_seconds_between_txn"] = avg_gap

    velocity_ratio = df["seconds_since_prev"] / df["avg_seconds_between_txn"]
    # No usable history yet -> treat as "typical" cadence (ratio 1) rather
    # than an undefined value.
    df["velocity_ratio"] = velocity_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    df["txn_count_last_1h"] = _rolling_trailing_count(df, "1h")
    df["txn_count_last_24h"] = _rolling_trailing_count(df, "24h")
    df["amount_sum_last_1h"] = _rolling_trailing_sum(df, "amount_clp", "1h")

    return df


def _rolling_trailing_count(df: pd.DataFrame, window: str) -> pd.Series:
    def _count(g: pd.DataFrame) -> pd.Series:
        s = g.set_index("timestamp")["amount_clp"]
        counts = s.rolling(window).count() - 1.0  # exclude the row itself
        return pd.Series(counts.to_numpy(), index=g.index)

    return df.groupby("customer_id", group_keys=False)[["timestamp", "amount_clp"]].apply(_count)


def _rolling_trailing_sum(df: pd.DataFrame, value_col: str, window: str) -> pd.Series:
    def _sum(g: pd.DataFrame) -> pd.Series:
        s = g.set_index("timestamp")[value_col]
        rolling_sum = s.rolling(window).sum().to_numpy() - s.to_numpy()  # exclude self
        return pd.Series(rolling_sum, index=g.index)

    return df.groupby("customer_id", group_keys=False)[["timestamp", value_col]].apply(_sum)
