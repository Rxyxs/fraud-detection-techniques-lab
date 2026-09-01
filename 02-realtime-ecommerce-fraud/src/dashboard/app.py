"""Streamlit monitoring dashboard: replays the held-out test split in
chronological order as if it were a live transaction feed, scoring each
transaction with the trained ensemble and raising alerts above the tuned
cost-sensitive threshold.

Run with: streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.generate_transactions import time_based_split  # noqa: E402
from src.features.build_features import NUMERIC_FEATURE_COLUMNS  # noqa: E402
from src.models.autoencoder import FraudAutoencoder, reconstruction_error  # noqa: E402

MODELS_DIR = ROOT / "outputs" / "models"
PLOTS_DIR = ROOT / "outputs" / "plots"
PROCESSED_PATH = ROOT / "data" / "processed" / "features.parquet"
FULL_FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + ["autoencoder_score"]

st.set_page_config(page_title="Detección de Fraude Financiero Chile", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_artifacts():
    with open(MODELS_DIR / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")

    ae_model = FraudAutoencoder(
        n_features=metadata["autoencoder_n_features"], latent_dim=metadata["autoencoder_latent_dim"]
    )
    ae_model.load_state_dict(torch.load(MODELS_DIR / "autoencoder.pt", map_location="cpu"))
    ae_model.eval()

    cat_model = CatBoostClassifier()
    cat_model.load_model(str(MODELS_DIR / "catboost_fraud.cbm"))
    xgb_model = XGBClassifier()
    xgb_model.load_model(str(MODELS_DIR / "xgboost_fraud.json"))

    return metadata, scaler, ae_model, cat_model, xgb_model


@st.cache_data
def load_scored_test_split():
    features = pd.read_parquet(PROCESSED_PATH)
    _, _, test_df = time_based_split(features)
    test_df = test_df.reset_index(drop=True)

    metadata, scaler, ae_model, cat_model, xgb_model = load_artifacts()

    X_numeric = test_df[NUMERIC_FEATURE_COLUMNS].to_numpy()
    X_scaled = scaler.transform(X_numeric)
    test_df["autoencoder_score"] = reconstruction_error(ae_model, X_scaled)

    X_full = test_df[FULL_FEATURE_COLUMNS]
    proba_cat = cat_model.predict_proba(X_full)[:, 1]
    proba_xgb = xgb_model.predict_proba(X_full)[:, 1]
    test_df["fraud_probability"] = (proba_cat + proba_xgb) / 2.0

    return test_df, metadata["decision_threshold"]


def kpi_row(df: pd.DataFrame, threshold: float):
    scored = df.copy()
    scored["alert"] = scored["fraud_probability"] >= threshold
    n_total = len(scored)
    n_alerts = int(scored["alert"].sum())
    n_true_fraud = int(scored["is_fraud"].sum())
    n_correct_alerts = int(((scored["alert"]) & (scored["is_fraud"] == 1)).sum())
    precision = n_correct_alerts / n_alerts if n_alerts else 0.0
    recall = n_correct_alerts / n_true_fraud if n_true_fraud else 0.0
    amount_flagged = scored.loc[scored["alert"], "amount_clp"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transacciones evaluadas", f"{n_total:,}")
    c2.metric("Alertas emitidas", f"{n_alerts:,}")
    c3.metric("Monto CLP en alertas", f"${amount_flagged:,.0f}")
    c4.metric("Precisión", f"{precision:.1%}")
    c5.metric("Recall", f"{recall:.1%}")


def main():
    st.title("🛡️ Sistema de Detección de Fraude Financiero — Chile")
    st.caption(
        "Réplica en vivo del conjunto de prueba (held-out, orden cronológico real) "
        "puntuado por el ensemble CatBoost + XGBoost cost-sensitive, con pre-filtro "
        "de anomalías vía Autoencoder (PyTorch)."
    )

    test_df, trained_threshold = load_scored_test_split()

    st.sidebar.header("Configuración")
    threshold = st.sidebar.slider(
        "Umbral de decisión (fraud_probability >= umbral)",
        min_value=0.0, max_value=1.0, value=float(trained_threshold), step=0.005,
    )
    st.sidebar.caption(f"Umbral óptimo entrenado (costo-sensible): {trained_threshold:.3f}")

    categories = ["(todas)"] + sorted(test_df["merchant_category"].unique().tolist())
    category_filter = st.sidebar.selectbox("Categoría de comercio", categories)

    view = test_df if category_filter == "(todas)" else test_df[test_df["merchant_category"] == category_filter]

    kpi_row(view, threshold)

    tab_live, tab_map, tab_model = st.tabs(["📡 Feed en vivo", "🗺️ Geolocalización", "📊 Rendimiento del modelo"])

    with tab_live:
        st.subheader("Simulación de transacciones en tiempo real")
        n_replay = st.slider("Transacciones a simular", 20, 300, 100, step=10)
        speed = st.select_slider("Velocidad", options=["Lenta", "Normal", "Rápida"], value="Normal")
        delay = {"Lenta": 0.15, "Normal": 0.05, "Rápida": 0.01}[speed]

        if st.button("▶ Iniciar simulación"):
            replay = view.sort_values("timestamp").head(n_replay).reset_index(drop=True)
            table_slot = st.empty()
            alert_slot = st.empty()
            progress = st.progress(0.0)
            log_rows = []
            n_alerts_seen = 0

            for i, row in replay.iterrows():
                is_alert = row["fraud_probability"] >= threshold
                n_alerts_seen += int(is_alert)
                log_rows.append({
                    "hora": row["timestamp"],
                    "cliente": row["customer_id"],
                    "monto_clp": f"${row['amount_clp']:,.0f}",
                    "categoria": row["merchant_category"],
                    "prob_fraude": f"{row['fraud_probability']:.3f}",
                    "alerta": "🚨 FRAUDE" if is_alert else "✅ OK",
                })
                table_slot.dataframe(pd.DataFrame(log_rows[-15:]), width="stretch")
                if is_alert:
                    alert_slot.error(
                        f"🚨 Transacción {row['transaction_id']} — cliente {row['customer_id']} — "
                        f"${row['amount_clp']:,.0f} CLP — prob. fraude {row['fraud_probability']:.1%}"
                    )
                progress.progress((i + 1) / len(replay))
                time.sleep(delay)

            st.success(f"Simulación completa: {n_alerts_seen} alertas de {len(replay)} transacciones.")

    with tab_map:
        st.subheader("Ubicación geográfica de transacciones")
        sample = view.sample(min(3000, len(view)), random_state=42) if len(view) > 3000 else view
        sample = sample.copy()
        sample["estado"] = np.where(sample["is_fraud"] == 1, "Fraude real", "Legítima")
        fig = px.scatter_mapbox(
            sample, lat="latitude", lon="longitude", color="estado",
            color_discrete_map={"Fraude real": "#c0392b", "Legítima": "#2c3e50"},
            size="amount_clp", size_max=15, zoom=3.2, height=600,
            hover_data=["transaction_id", "amount_clp", "fraud_probability"],
            center={"lat": -35.5, "lon": -71.5},
        )
        fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)

    with tab_model:
        st.subheader("Métricas del modelo (conjunto de prueba completo)")
        col1, col2, col3 = st.columns(3)
        for col, name in zip(
            (col1, col2, col3),
            ("precision_recall_curve.png", "confusion_matrix.png", "feature_importance.png"),
        ):
            path = PLOTS_DIR / name
            if path.exists():
                col.image(str(path), width="stretch")

        report_path = ROOT / "outputs" / "reports" / "training_report.json"
        if report_path.exists():
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
            st.json(report["ensemble"])


if __name__ == "__main__":
    main()
