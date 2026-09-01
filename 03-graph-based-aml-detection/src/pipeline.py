"""Pipeline end-to-end: datos sinteticos -> grafo -> features -> ensamble
no supervisado -> alertas y reporte de evaluacion.

Uso:
    python -m src.pipeline
    python -m src.pipeline --regenerar-datos --contaminacion 0.04
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import networkx as nx
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.synthetic_generator import END_DATE, OUT_DIR as DATA_DIR, UMBRAL_ESTRUCTURACION_CLP
from data.synthetic_generator import main as generar_datos
from src.anomaly.ensemble_detector import evaluate_against_ground_truth, run_ensemble
from src.anomaly.transaction_scorer import (
    build_transaction_features,
    evaluate_tx_against_ground_truth,
    run_transaction_ensemble,
    train_production_model,
)
from src.graph.graph_features import build_feature_table
from src.graph.network_builder import build_multidigraph, build_weighted_digraph, ensure_all_accounts_present
from src.graph.temporal_graph_exporter import export_temporal_graph

OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"


def cargar_datos(regenerar: bool) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if regenerar or not (DATA_DIR / "transfers.parquet").exists():
        print("Generando datos sinteticos...")
        generar_datos()
    accounts = pl.read_parquet(DATA_DIR / "accounts.parquet")
    transfers = pl.read_parquet(DATA_DIR / "transfers.parquet")
    ground_truth = pl.read_parquet(DATA_DIR / "ground_truth.parquet")
    return accounts, transfers, ground_truth


def ejecutar_pipeline(
    regenerar_datos: bool = False, contaminacion: float = 0.05, contaminacion_tx: float = 0.01
) -> dict:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    accounts, transfers, ground_truth = cargar_datos(regenerar_datos)
    print(f"Cuentas: {accounts.height:,} | Transferencias: {transfers.height:,}")

    print("Construyendo grafo dirigido de transferencias...")
    graph = build_weighted_digraph(transfers)
    graph = ensure_all_accounts_present(graph, accounts["account_id"].to_list())
    print(f"Nodos: {graph.number_of_nodes():,} | Aristas: {graph.number_of_edges():,}")

    print("Calculando metricas de grafo y features transaccionales (nivel cuenta)...")
    features = build_feature_table(graph, transfers, accounts, UMBRAL_ESTRUCTURACION_CLP, END_DATE)

    print(f"Ejecutando ensamble no supervisado (IForest + COPOD + ECOD), contaminacion={contaminacion}...")
    scored = run_ensemble(features, contamination=contaminacion)

    metricas = evaluate_against_ground_truth(scored, ground_truth)

    alertas = scored.filter(pl.col("alerta")).join(
        ground_truth, on="account_id", how="left"
    ).sort("score_ensamble", descending=True)

    scored.write_parquet(OUTPUTS_DIR / "cuentas_con_score.parquet")
    alertas.write_csv(OUTPUTS_DIR / "alertas_uaf.csv")
    nx.write_graphml(graph, OUTPUTS_DIR / "red_transferencias.graphml")

    multigraph = build_multidigraph(transfers)
    import pickle
    with open(OUTPUTS_DIR / "multigrafo_transferencias.gpickle", "wb") as f:
        pickle.dump(multigraph, f)

    print(f"\nCalculando features por transaccion individual (nivel transferencia)...")
    tx_features = build_transaction_features(transfers, features, UMBRAL_ESTRUCTURACION_CLP)
    print(f"Ejecutando ensamble no supervisado por transaccion, contaminacion={contaminacion_tx}...")
    scored_tx = run_transaction_ensemble(tx_features, contamination=contaminacion_tx)
    metricas_tx = evaluate_tx_against_ground_truth(scored_tx)
    scored_tx.write_parquet(OUTPUTS_DIR / "transacciones_con_score.parquet")
    scored_tx.filter(pl.col("alerta_tx")).write_csv(OUTPUTS_DIR / "alertas_transacciones.csv")

    print("Entrenando y persistiendo modelo de produccion (IsolationForest + SHAP) para la API...")
    info_modelo = train_production_model(tx_features, MODELS_DIR, contamination=contaminacion_tx)
    print(f"  Modelo guardado en {MODELS_DIR} | umbral de alerta: {info_modelo['umbral_alerta']:.4f}")

    print("Exportando grafo temporal y evolucion diaria de la red...")
    info_temporal = export_temporal_graph(transfers, OUTPUTS_DIR)
    print(f"  {info_temporal['n_nodos_temporales']:,} nodos, {info_temporal['n_aristas_temporales']:,} "
          f"aristas temporales, {info_temporal['n_dias']} dias")

    reporte = OUTPUTS_DIR / "evaluation_report.md"
    with open(reporte, "w", encoding="utf-8") as f:
        f.write("# Reporte de evaluacion - motor de deteccion de anomalias AML\n\n")
        f.write("Evaluacion contra verdad terreno sintetica (inyectada en el generador). ")
        f.write("Esta etiqueta NO se usa para entrenar el ensamble; solo para validar su desempenio.\n\n")
        f.write("## Nivel cuenta (agregado)\n\n")
        for k, v in metricas.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Nivel transaccion individual\n\n")
        for k, v in metricas_tx.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Distribucion de tipologias entre las alertas emitidas (nivel cuenta)\n\n")
        dist = alertas.group_by("tipologia_real").agg(pl.len().alias("n")).sort("n", descending=True)
        f.write(dist.to_pandas().to_markdown(index=False))
        f.write("\n\n## Distribucion de tipologias entre las alertas emitidas (nivel transaccion)\n\n")
        dist_tx = (
            scored_tx.filter(pl.col("alerta_tx"))
            .group_by("tipologia").agg(pl.len().alias("n")).sort("n", descending=True)
        )
        f.write(dist_tx.to_pandas().to_markdown(index=False))
        f.write("\n")

    print("\n=== METRICAS DE EVALUACION - NIVEL CUENTA (solo validacion) ===")
    for k, v in metricas.items():
        print(f"  {k}: {v}")
    print("\n=== METRICAS DE EVALUACION - NIVEL TRANSACCION (solo validacion) ===")
    for k, v in metricas_tx.items():
        print(f"  {k}: {v}")
    print(f"\nArtefactos escritos en: {OUTPUTS_DIR}")

    return {**metricas, **metricas_tx}


def main():
    parser = argparse.ArgumentParser(description="Pipeline AML de deteccion de anomalias no supervisado")
    parser.add_argument("--regenerar-datos", action="store_true", help="Fuerza regenerar los datos sinteticos")
    parser.add_argument("--contaminacion", type=float, default=0.05, help="Proporcion esperada de cuentas anomalas")
    parser.add_argument(
        "--contaminacion-tx", type=float, default=0.01, help="Proporcion esperada de transacciones anomalas"
    )
    args = parser.parse_args()
    ejecutar_pipeline(
        regenerar_datos=args.regenerar_datos,
        contaminacion=args.contaminacion,
        contaminacion_tx=args.contaminacion_tx,
    )


if __name__ == "__main__":
    main()
