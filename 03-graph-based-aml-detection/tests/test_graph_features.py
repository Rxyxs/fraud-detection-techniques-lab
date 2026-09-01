import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph.graph_features import build_feature_table, compute_graph_metrics
from src.graph.network_builder import build_weighted_digraph, ensure_all_accounts_present

BASE_TS = datetime(2026, 1, 1)


def _toy_transfers():
    # A -> B -> C: B es una cuenta puente perfecta (recibe y reenvia el mismo monto)
    # D es un hub que envia a muchas cuentas distintas
    # E es una cuenta aislada (sin transferencias)
    rows = [
        {"origen": "A", "destino": "B", "monto_clp": 1_000_000.0, "timestamp": BASE_TS},
        {"origen": "B", "destino": "C", "monto_clp": 1_000_000.0, "timestamp": BASE_TS + timedelta(hours=1)},
        {"origen": "D", "destino": "A", "monto_clp": 50_000.0, "timestamp": BASE_TS},
        {"origen": "D", "destino": "B", "monto_clp": 50_000.0, "timestamp": BASE_TS},
        {"origen": "D", "destino": "C", "monto_clp": 50_000.0, "timestamp": BASE_TS},
        {"origen": "D", "destino": "F", "monto_clp": 50_000.0, "timestamp": BASE_TS},
    ]
    return pl.DataFrame(rows)


def test_pass_through_ratio_flags_bridge_account():
    transfers = _toy_transfers()
    graph = build_weighted_digraph(transfers)
    graph = ensure_all_accounts_present(graph, ["A", "B", "C", "D", "E", "F"])
    feats = compute_graph_metrics(graph).sort("account_id")

    b_row = feats.filter(pl.col("account_id") == "B").row(0, named=True)
    assert b_row["ratio_paso"] > 0.95  # B recibe y reenvia (casi) el mismo monto


def test_isolated_account_has_zero_degree():
    transfers = _toy_transfers()
    graph = build_weighted_digraph(transfers)
    graph = ensure_all_accounts_present(graph, ["A", "B", "C", "D", "E", "F"])
    feats = compute_graph_metrics(graph).sort("account_id")

    e_row = feats.filter(pl.col("account_id") == "E").row(0, named=True)
    assert e_row["grado_entrada"] == 0
    assert e_row["grado_salida"] == 0
    assert e_row["pagerank"] >= 0


def test_hub_account_has_highest_out_degree():
    transfers = _toy_transfers()
    graph = build_weighted_digraph(transfers)
    graph = ensure_all_accounts_present(graph, ["A", "B", "C", "D", "E", "F"])
    feats = compute_graph_metrics(graph)

    top_out = feats.sort("grado_salida", descending=True).row(0, named=True)
    assert top_out["account_id"] == "D"
    assert top_out["grado_salida"] == 4


def test_feature_table_covers_all_accounts_no_nulls():
    transfers = _toy_transfers()
    accounts = pl.DataFrame(
        {
            "account_id": ["A", "B", "C", "D", "E", "F"],
            "tipo_cliente": ["Persona Natural"] * 6,
            "banco": ["Banco X"] * 6,
            "region": ["Region Metropolitana"] * 6,
            "fecha_apertura": [BASE_TS - timedelta(days=365)] * 6,
        }
    )
    graph = build_weighted_digraph(transfers)
    graph = ensure_all_accounts_present(graph, accounts["account_id"].to_list())

    features = build_feature_table(graph, transfers, accounts, umbral_estructuracion=5_000_000, fecha_referencia=BASE_TS)

    assert features.height == 6
    assert features.null_count().to_numpy().sum() == 0
