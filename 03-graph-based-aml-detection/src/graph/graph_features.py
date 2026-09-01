"""Extraccion de metricas de teoria de grafos y features transaccionales
por cuenta, a partir de la red de transferencias TEF.

Metricas de grafo (NetworkX), sobre el grafo dirigido y ponderado por monto:
    - grado de entrada/salida y su version ponderada (monto total)
    - PageRank (importancia dentro del flujo de fondos)
    - coeficiente de agrupamiento local (proyeccion no dirigida)
    - centralidad de intermediacion aproximada (deteccion de "puentes")
    - reciprocidad local (fraccion de contrapartes mutuas)
    - tamanio de la componente fuertemente conexa (ciclos de fondos)
    - ratio de paso ("pass-through"): min(monto_in, monto_out) / max(...),
      cercano a 1 en cuentas mula que reenvian casi todo lo que reciben

Features transaccionales (Polars), sobre la tabla de transferencias:
    - conteos y estadisticas de monto enviado/recibido
    - conteo de transferencias cercanas al umbral de fraccionamiento
    - proporcion de operaciones fuera de horario habil
    - "burst score": maximo de transferencias recibidas en 24h moviles
    - antiguedad de la cuenta en dias
"""

from __future__ import annotations

from datetime import timedelta

import networkx as nx
import polars as pl

NIGHT_HOURS = set(range(22, 24)) | set(range(0, 6))


def compute_graph_metrics(graph: nx.DiGraph, betweenness_k: int = 400, seed: int = 42) -> pl.DataFrame:
    nodes = list(graph.nodes)
    n = len(nodes)
    k = min(betweenness_k, n) if n > 2 else None

    pagerank = nx.pagerank(graph, weight="weight") if graph.number_of_edges() else {node: 0.0 for node in nodes}
    clustering = nx.clustering(graph.to_undirected())
    betweenness = nx.betweenness_centrality(graph, k=k, seed=seed, normalized=True) if k else {node: 0.0 for node in nodes}
    sccs = nx.strongly_connected_components(graph)
    scc_size = {}
    for comp in sccs:
        size = len(comp)
        for node in comp:
            scc_size[node] = size

    rows = []
    for node in nodes:
        in_deg = graph.in_degree(node)
        out_deg = graph.out_degree(node)
        in_weight = sum(d["monto_total_clp"] for _, _, d in graph.in_edges(node, data=True))
        out_weight = sum(d["monto_total_clp"] for _, _, d in graph.out_edges(node, data=True))
        pred = set(graph.predecessors(node))
        succ = set(graph.successors(node))
        mutual = len(pred & succ)
        total_deg = in_deg + out_deg
        reciprocidad = (2 * mutual / total_deg) if total_deg > 0 else 0.0
        max_flujo = max(in_weight, out_weight)
        pass_through_ratio = (min(in_weight, out_weight) / max_flujo) if max_flujo > 0 else 0.0

        rows.append(
            {
                "account_id": node,
                "grado_entrada": in_deg,
                "grado_salida": out_deg,
                "monto_entrada_total": in_weight,
                "monto_salida_total": out_weight,
                "pagerank": pagerank.get(node, 0.0),
                "coef_agrupamiento": clustering.get(node, 0.0),
                "centralidad_intermediacion": betweenness.get(node, 0.0),
                "reciprocidad_local": reciprocidad,
                "tamanio_componente_conexa": scc_size.get(node, 1),
                "ratio_paso": pass_through_ratio,
            }
        )
    return pl.DataFrame(rows)


def _burst_scores(transfers: pl.DataFrame) -> dict[str, int]:
    """Maximo de transferencias recibidas por cuenta en cualquier ventana
    movil de 24 horas (senial de rafaga / fan-in de cuenta nueva)."""
    scores: dict[str, int] = {}
    window = timedelta(hours=24)
    for key, group in transfers.select(["destino", "timestamp"]).sort("timestamp").group_by("destino"):
        account_id = key[0] if isinstance(key, tuple) else key
        ts = sorted(group["timestamp"].to_list())
        max_count, left = 0, 0
        for right in range(len(ts)):
            while ts[right] - ts[left] > window:
                left += 1
            max_count = max(max_count, right - left + 1)
        scores[account_id] = max_count
    return scores


def compute_transactional_features(
    transfers: pl.DataFrame,
    accounts: pl.DataFrame,
    umbral_estructuracion: float,
    fecha_referencia,
) -> pl.DataFrame:
    enviadas = (
        transfers.group_by("origen")
        .agg(
            pl.len().alias("n_enviadas"),
            pl.col("monto_clp").mean().alias("monto_prom_enviado"),
            pl.col("monto_clp").std().fill_null(0.0).alias("monto_std_enviado"),
            pl.col("monto_clp").max().alias("monto_max_enviado"),
            (
                (pl.col("monto_clp") >= 0.85 * umbral_estructuracion)
                & (pl.col("monto_clp") < umbral_estructuracion)
            ).sum().alias("n_cercanas_umbral"),
        )
        .rename({"origen": "account_id"})
    )

    recibidas = (
        transfers.group_by("destino")
        .agg(
            pl.len().alias("n_recibidas"),
            pl.col("monto_clp").mean().alias("monto_prom_recibido"),
            pl.col("monto_clp").std().fill_null(0.0).alias("monto_std_recibido"),
            pl.col("monto_clp").max().alias("monto_max_recibido"),
        )
        .rename({"destino": "account_id"})
    )

    con_hora = transfers.with_columns(pl.col("timestamp").dt.hour().alias("hora"))
    nocturnas_env = con_hora.group_by("origen").agg(
        pl.col("hora").is_in(list(NIGHT_HOURS)).sum().alias("n_nocturnas_origen"),
        pl.len().alias("n_total_origen"),
    ).rename({"origen": "account_id"})
    nocturnas_dest = con_hora.group_by("destino").agg(
        pl.col("hora").is_in(list(NIGHT_HOURS)).sum().alias("n_nocturnas_destino"),
        pl.len().alias("n_total_destino"),
    ).rename({"destino": "account_id"})

    burst = _burst_scores(transfers)
    burst_df = pl.DataFrame(
        {"account_id": list(burst.keys()), "burst_score_24h": list(burst.values())}
    )

    antiguedad = accounts.select(
        "account_id",
        ((pl.lit(fecha_referencia) - pl.col("fecha_apertura")).dt.total_days()).alias("antiguedad_dias"),
    )

    features = (
        accounts.select("account_id")
        .join(enviadas, on="account_id", how="left")
        .join(recibidas, on="account_id", how="left")
        .join(nocturnas_env, on="account_id", how="left")
        .join(nocturnas_dest, on="account_id", how="left")
        .join(burst_df, on="account_id", how="left")
        .join(antiguedad, on="account_id", how="left")
        .fill_null(0)
    )

    features = features.with_columns(
        (
            (pl.col("n_nocturnas_origen") + pl.col("n_nocturnas_destino"))
            / (pl.col("n_total_origen") + pl.col("n_total_destino")).clip(lower_bound=1)
        ).alias("ratio_nocturno")
    ).drop(["n_nocturnas_origen", "n_nocturnas_destino", "n_total_origen", "n_total_destino"])

    return features


def build_feature_table(
    graph: nx.DiGraph,
    transfers: pl.DataFrame,
    accounts: pl.DataFrame,
    umbral_estructuracion: float,
    fecha_referencia,
) -> pl.DataFrame:
    graph_feats = compute_graph_metrics(graph)
    tx_feats = compute_transactional_features(transfers, accounts, umbral_estructuracion, fecha_referencia)
    return accounts.select("account_id", "tipo_cliente", "banco", "region").join(
        graph_feats, on="account_id", how="left"
    ).join(tx_feats, on="account_id", how="left").fill_null(0)
