import sqlite3

import pandas as pd
import pytest

from src.sql_reports import DB_PATH, SQL_PATH, VIEWS


@pytest.mark.skipif(not DB_PATH.exists(), reason="requires outputs/fraud.sqlite from python -m src.pipeline")
def test_all_views_apply_and_return_rows():
    con = sqlite3.connect(str(DB_PATH))
    con.executescript(SQL_PATH.read_text(encoding="utf-8"))
    for view in VIEWS:
        df = pd.read_sql(f"SELECT * FROM {view}", con)
        assert len(df) > 0, f"{view} returned no rows"
    con.close()


@pytest.mark.skipif(not DB_PATH.exists(), reason="requires outputs/fraud.sqlite from python -m src.pipeline")
def test_decile_view_fraud_rate_increases_monotonically_ish():
    """No estrictamente monotonico por definicion, pero el decil 1 (menor
    score) deberia tener una tasa de fraude mucho menor que el decil 10."""
    con = sqlite3.connect(str(DB_PATH))
    con.executescript(SQL_PATH.read_text(encoding="utf-8"))
    df = pd.read_sql("SELECT * FROM v_catboost_decile_performance ORDER BY decile", con)
    con.close()
    assert df.iloc[0]["actual_fraud_rate"] < df.iloc[-1]["actual_fraud_rate"]
