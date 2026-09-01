"""Exportacion de la red de transferencias como grafo temporal.

``network_builder.build_weighted_digraph`` colapsa toda la ventana de 90
dias en un unico grafo estatico, lo que aplana exactamente la senial que
distingue a una cadena de "cuentas puente" (layering): fondos que entran y
salen de una cuenta en cuestion de horas. Este modulo conserva el eje
temporal de dos formas complementarias:

    - ``build_temporal_multigraph``: un ``nx.MultiDiGraph`` con una arista
      por transferencia, cada una con atributo ``timestamp`` (formato ISO,
      requerido por GraphML) -- permite reconstruir el estado de la red en
      cualquier instante o inspeccionar la secuencia exacta de una cadena.
    - ``daily_network_evolution``: una serie temporal (una fila por dia) de
      metricas agregadas de la red -- nodos activos, aristas, monto total y
      grado promedio -- para visualizar como evoluciona la topologia dia a
      dia en vez de verla como una sola foto fija.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import polars as pl


def build_temporal_multigraph(transfers: pl.DataFrame) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for row in transfers.iter_rows(named=True):
        graph.add_edge(
            row["origen"],
            row["destino"],
            transfer_id=row["transfer_id"],
            monto_clp=float(row["monto_clp"]),
            timestamp=row["timestamp"].isoformat(),
            tipologia=row["tipologia"] or "normal",
        )
    return graph


def daily_network_evolution(transfers: pl.DataFrame) -> pl.DataFrame:
    """Metricas de la red por dia calendario: cuentas activas (origen o
    destino), transferencias, monto total y grado promedio (aristas*2 /
    nodos activos) -- una radiografia de como crece y se contrae la red
    dia a dia, en vez de un unico grafo agregado sobre toda la ventana."""
    con_dia = transfers.with_columns(pl.col("timestamp").dt.date().alias("dia"))

    por_dia = con_dia.group_by("dia").agg(
        pl.len().alias("n_transferencias"),
        pl.col("monto_clp").sum().alias("monto_total_clp"),
        pl.concat_list([pl.col("origen"), pl.col("destino")]).alias("_participantes"),
    ).sort("dia")

    filas = []
    for row in por_dia.iter_rows(named=True):
        participantes = set()
        for par in row["_participantes"]:
            participantes.update(par)
        n_nodos = len(participantes)
        filas.append({
            "dia": row["dia"],
            "cuentas_activas": n_nodos,
            "n_transferencias": row["n_transferencias"],
            "monto_total_clp": row["monto_total_clp"],
            "grado_promedio": (2 * row["n_transferencias"] / n_nodos) if n_nodos else 0.0,
        })
    return pl.DataFrame(filas)


def export_temporal_graph(transfers: pl.DataFrame, outputs_dir: Path) -> dict:
    outputs_dir.mkdir(exist_ok=True)

    grafo_temporal = build_temporal_multigraph(transfers)
    nx.write_graphml(grafo_temporal, outputs_dir / "red_temporal.graphml")

    evolucion = daily_network_evolution(transfers)
    evolucion.write_csv(outputs_dir / "evolucion_red_diaria.csv")

    return {
        "n_aristas_temporales": grafo_temporal.number_of_edges(),
        "n_nodos_temporales": grafo_temporal.number_of_nodes(),
        "n_dias": evolucion.height,
    }
