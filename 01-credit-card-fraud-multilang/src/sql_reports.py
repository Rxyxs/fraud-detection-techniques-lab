"""Aplica sql/analytical_views.sql sobre outputs/fraud.sqlite y exporta el
resultado de cada vista -- el trabajo analitico real vive en el .sql, esto
solo lo ejecuta y persiste los resultados.

    python -m src.sql_reports
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "outputs" / "fraud.sqlite"
SQL_PATH = ROOT / "sql" / "analytical_views.sql"
REPORTS_DIR = ROOT / "outputs" / "reports"

VIEWS = ["v_model_ranking", "v_catboost_decile_performance", "v_model_disagreement", "v_executive_summary"]


def main() -> None:
    con = sqlite3.connect(str(DB_PATH))
    con.executescript(SQL_PATH.read_text(encoding="utf-8"))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for view in VIEWS:
        df = pd.read_sql(f"SELECT * FROM {view}", con)
        df.to_csv(REPORTS_DIR / f"{view}.csv", index=False)
        print(f"\n=== {view} ===")
        print(df.head(10).to_string(index=False))

    con.close()
    print(f"\nGuardado en: {REPORTS_DIR}/v_*.csv")


if __name__ == "__main__":
    main()
