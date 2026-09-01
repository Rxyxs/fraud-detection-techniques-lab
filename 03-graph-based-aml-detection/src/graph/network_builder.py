"""Construccion de la red de transferencias TEF como grafo dirigido.

Se construyen dos representaciones a partir de la misma tabla de
transferencias:

* ``build_multidigraph``: un ``nx.MultiDiGraph`` que conserva cada
  transferencia como una arista independiente (con monto y timestamp).
  Util para visualizacion detallada (PyVis) y para inspeccionar el detalle
  de una relacion origen-destino especifica.
* ``build_weighted_digraph``: un ``nx.DiGraph`` agregado (una arista por
  par origen-destino, con monto total y conteo) usado para calcular
  metricas de centralidad de forma eficiente sobre ~2.000 nodos.
"""

from __future__ import annotations

import networkx as nx
import polars as pl


def build_multidigraph(transfers: pl.DataFrame) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for row in transfers.iter_rows(named=True):
        graph.add_edge(
            row["origen"],
            row["destino"],
            transfer_id=row["transfer_id"],
            monto_clp=row["monto_clp"],
            timestamp=row["timestamp"],
            tipologia=row["tipologia"],
        )
    return graph


def build_weighted_digraph(transfers: pl.DataFrame) -> nx.DiGraph:
    agg = (
        transfers.group_by(["origen", "destino"])
        .agg(
            pl.len().alias("n_transferencias"),
            pl.col("monto_clp").sum().alias("monto_total_clp"),
        )
        .sort(["origen", "destino"])
    )
    graph = nx.DiGraph()
    for row in agg.iter_rows(named=True):
        graph.add_edge(
            row["origen"],
            row["destino"],
            weight=float(row["monto_total_clp"]),
            n_transferencias=int(row["n_transferencias"]),
            monto_total_clp=float(row["monto_total_clp"]),
        )
    return graph


def ensure_all_accounts_present(graph: nx.DiGraph, account_ids: list[str]) -> nx.DiGraph:
    """Agrega como nodos aislados las cuentas sin transferencias (grado 0)."""
    graph.add_nodes_from(account_ids)
    return graph
