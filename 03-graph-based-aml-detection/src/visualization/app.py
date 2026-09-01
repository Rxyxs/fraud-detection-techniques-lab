"""Dashboard Streamlit: exploracion interactiva de alertas AML y su red de
transferencias (grafo PyVis embebido).

Uso:
    streamlit run src/visualization/app.py

Requiere haber corrido antes el pipeline (``python -m src.pipeline``), que
deja los artefactos usados aqui en ``outputs/``.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import networkx as nx
import plotly.express as px
import polars as pl
import streamlit as st
from pyvis.network import Network

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUTPUTS_DIR = ROOT / "outputs"

st.set_page_config(page_title="Motor de Deteccion de Anomalias AML - Chile", layout="wide")

TIPOLOGIA_COLORES = {
    "pitufeo": "#e74c3c",
    "cuenta_puente": "#e67e22",
    "rafaga_cuenta_nueva": "#9b59b6",
    "monto_inusual": "#f1c40f",
    "normal": "#2ecc71",
}


@st.cache_data
def cargar_datos():
    cuentas = pl.read_parquet(OUTPUTS_DIR / "cuentas_con_score.parquet")
    alertas = pl.read_csv(OUTPUTS_DIR / "alertas_uaf.csv")
    with open(OUTPUTS_DIR / "multigrafo_transferencias.gpickle", "rb") as f:
        multigrafo = pickle.load(f)
    return cuentas, alertas, multigrafo


def construir_ego_network_html(multigrafo: nx.MultiDiGraph, cuentas_foco: list[str], scores: dict, radio: int = 1) -> str:
    nodos = set(cuentas_foco)
    frontera = set(cuentas_foco)
    for _ in range(radio):
        nueva_frontera = set()
        for nodo in frontera:
            if nodo in multigrafo:
                nueva_frontera |= set(multigrafo.predecessors(nodo)) | set(multigrafo.successors(nodo))
        nodos |= nueva_frontera
        frontera = nueva_frontera

    subgrafo = multigrafo.subgraph(nodos)
    net = Network(height="600px", width="100%", directed=True, bgcolor="#0e1117", font_color="white")
    net.barnes_hut(gravity=-4000, central_gravity=0.2, spring_length=120)

    for nodo in subgrafo.nodes:
        score = scores.get(nodo, 0.0)
        es_foco = nodo in cuentas_foco
        size = 18 + min(score, 10) * 2.5
        color = "#e74c3c" if es_foco else "#3498db"
        net.add_node(
            nodo, label=nodo, size=size, color=color,
            title=f"{nodo}<br>score ensamble: {score:.2f}",
        )

    for origen, destino, datos in subgrafo.edges(data=True):
        monto = datos.get("monto_clp", 0)
        tipologia = datos.get("tipologia", "normal")
        net.add_edge(
            origen, destino,
            value=max(1, monto / 1_000_000),
            title=f"${monto:,.0f} CLP - {tipologia}",
            color=TIPOLOGIA_COLORES.get(tipologia, "#7f8c8d"),
        )

    return net.generate_html(notebook=False)


def main():
    st.title("🔎 Motor de Deteccion de Anomalias AML — Red TEF Chile")
    st.caption(
        "Ensamble no supervisado (Isolation Forest + COPOD + ECOD) sobre metricas de grafo "
        "y transaccionales, alineado a tipologias de la UAF. Datos 100% sinteticos."
    )

    if not (OUTPUTS_DIR / "cuentas_con_score.parquet").exists():
        st.error(
            "No se encontraron artefactos en `outputs/`. Ejecuta primero "
            "`python -m src.pipeline` para generar los datos y correr el ensamble."
        )
        st.stop()

    cuentas, alertas, multigrafo = cargar_datos()
    scores_dict = dict(zip(cuentas["account_id"].to_list(), cuentas["score_ensamble"].to_list()))

    with st.sidebar:
        st.header("Filtros")
        bancos = ["Todos"] + sorted(cuentas["banco"].unique().to_list())
        banco_sel = st.selectbox("Banco", bancos)
        tipologias_disp = ["Todas"] + sorted(
            [t for t in alertas["tipologia_real"].unique().to_list() if t]
        )
        tipologia_sel = st.selectbox("Tipologia (verdad terreno, solo validacion)", tipologias_disp)
        top_n = st.slider("Top N cuentas a mostrar en tabla/grafo", 5, 100, 25)

    total_cuentas = cuentas.height
    total_transferencias = multigrafo.number_of_edges()
    n_alertas = alertas.height
    monto_total = sum(d.get("monto_clp", 0) for _, _, d in multigrafo.edges(data=True))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cuentas analizadas", f"{total_cuentas:,}")
    c2.metric("Transferencias TEF", f"{total_transferencias:,}")
    c3.metric("Alertas emitidas", f"{n_alertas:,}", f"{100 * n_alertas / total_cuentas:.1f}% del total")
    c4.metric("Monto total en la red", f"${monto_total / 1e9:,.1f} MM CLP")

    tab_alertas, tab_grafo, tab_tipologias = st.tabs(["📋 Alertas", "🕸️ Red interactiva", "📊 Tipologias"])

    filtro = alertas
    if banco_sel != "Todos":
        filtro = filtro.filter(pl.col("banco") == banco_sel)
    if tipologia_sel != "Todas":
        filtro = filtro.filter(pl.col("tipologia_real") == tipologia_sel)
    filtro = filtro.sort("score_ensamble", descending=True).head(top_n)

    with tab_alertas:
        st.subheader(f"Top {min(top_n, filtro.height)} cuentas con mayor score de anomalia")
        st.dataframe(
            filtro.select(
                "account_id", "banco", "region", "tipo_cliente", "grado_entrada", "grado_salida",
                "burst_score_24h", "ratio_paso", "n_cercanas_umbral", "score_ensamble", "tipologia_real",
            ).to_pandas(),
            width="stretch",
        )
        st.download_button(
            "Descargar alertas filtradas (CSV)",
            filtro.write_csv(),
            file_name="alertas_uaf_filtradas.csv",
        )

    with tab_grafo:
        st.subheader("Ego-red de las cuentas mas sospechosas")
        cuentas_foco = filtro["account_id"].to_list()[: min(10, top_n)]
        if cuentas_foco:
            html = construir_ego_network_html(multigrafo, cuentas_foco, scores_dict)
            st.iframe(html, height=620)
        else:
            st.info("No hay cuentas que cumplan los filtros seleccionados.")

    with tab_tipologias:
        st.subheader("Distribucion de tipologias detectadas entre las alertas")
        dist = (
            alertas.group_by("tipologia_real")
            .agg(pl.len().alias("n"))
            .with_columns(pl.col("tipologia_real").fill_null("sin_tipologia_conocida"))
            .sort("n", descending=True)
            .to_pandas()
        )
        fig = px.bar(dist, x="tipologia_real", y="n", color="tipologia_real",
                     color_discrete_map=TIPOLOGIA_COLORES, title="Alertas por tipologia (verdad terreno)")
        st.plotly_chart(fig, width="stretch")

        st.markdown(
            "> La columna `tipologia_real` proviene de la verdad terreno sintetica "
            "inyectada por el generador de datos; se muestra unicamente para "
            "validar el desempenio del ensamble no supervisado, no como input del modelo."
        )


if __name__ == "__main__":
    main()
