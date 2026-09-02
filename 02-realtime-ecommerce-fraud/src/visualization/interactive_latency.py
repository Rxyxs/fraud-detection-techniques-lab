"""Measures real /detect-fraud inference latency (same protocol as
tests/test_latency.py: 200 requests via FastAPI's in-process TestClient, a
mix of legit-shaped and fraud-shaped payloads) and renders an interactive
Plotly distribution (self-contained HTML) of both the model-only latency and
the full HTTP round-trip latency, with the 50ms production budget marked.

Requires having run `python run_pipeline.py` at least once (so the trained
model artifacts under outputs/models/ exist).

Usage:
    python -m src.visualization.interactive_latency
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import plotly.graph_objects as go
from fastapi.testclient import TestClient

from src.api.main import app

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "interactive"
N_REQUESTS = 200
LATENCY_BUDGET_MS = 50.0

LEGIT_PAYLOAD = {
    "transaction_id": "TXN_LAT_LEGIT",
    "customer_id": 123,
    "timestamp": "2026-03-15T14:30:00",
    "amount_clp": 25000,
    "merchant_category": "supermercado",
    "latitude": -33.45,
    "longitude": -70.66,
    "customer_state": {
        "last_latitude": -33.44,
        "last_longitude": -70.65,
        "last_timestamp": "2026-03-15T10:00:00",
        "home_latitude": -33.45,
        "home_longitude": -70.66,
        "avg_amount_clp": 20000,
        "std_amount_clp": 5000,
        "avg_seconds_between_txn": 86400,
        "txn_count_last_1h": 0,
        "txn_count_last_24h": 1,
        "amount_sum_last_1h": 0,
    },
}


def _fraud_payload(i: int) -> dict:
    payload = {**LEGIT_PAYLOAD, "transaction_id": f"TXN_LAT_FRAUD_{i}", "amount_clp": 950000}
    payload["latitude"] = -20.21
    payload["longitude"] = -70.15
    payload["customer_state"] = {
        **LEGIT_PAYLOAD["customer_state"],
        "last_timestamp": "2026-03-15T14:29:00",
        "txn_count_last_1h": 4,
    }
    return payload


def _percentile(values: list[float], pct: float) -> float:
    return sorted(values)[int(pct * (len(values) - 1))]


def main() -> None:
    model_latencies: list[float] = []
    e2e_latencies: list[float] = []

    with TestClient(app) as client:
        for i in range(N_REQUESTS):
            payload = (
                {**LEGIT_PAYLOAD, "transaction_id": f"TXN_LAT_{i}"}
                if i % 2 == 0
                else _fraud_payload(i)
            )
            start = time.perf_counter()
            r = client.post("/detect-fraud", json=payload)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            assert r.status_code == 200
            model_latencies.append(r.json()["model_latency_ms"])
            e2e_latencies.append(elapsed_ms)

    p50_model, p95_model = statistics.median(model_latencies), _percentile(model_latencies, 0.95)
    p50_e2e, p95_e2e = statistics.median(e2e_latencies), _percentile(e2e_latencies, 0.95)
    print(
        f"model-only: p50={p50_model:.2f}ms p95={p95_model:.2f}ms max={max(model_latencies):.2f}ms\n"
        f"end-to-end: p50={p50_e2e:.2f}ms p95={p95_e2e:.2f}ms max={max(e2e_latencies):.2f}ms"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=model_latencies,
            name=f"Model-only (p95={p95_model:.2f}ms)",
            marker_color="#3b7ddd",
            opacity=0.75,
            nbinsx=40,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=e2e_latencies,
            name=f"Full HTTP round-trip (p95={p95_e2e:.2f}ms)",
            marker_color="#e0824f",
            opacity=0.75,
            nbinsx=40,
        )
    )
    fig.add_vline(
        x=LATENCY_BUDGET_MS,
        line_dash="dash",
        line_color="#c0392b",
        annotation_text="50ms production budget",
        annotation_position="top right",
    )
    fig.update_layout(
        barmode="overlay",
        title=f"/detect-fraud inference latency — {N_REQUESTS} real requests via FastAPI TestClient"
        "<br><sup>Mix of legit-shaped and fraud-shaped payloads, one real run</sup>",
        xaxis_title="Latency (ms)",
        yaxis_title="Requests",
        template="plotly_white",
        width=1000,
        height=600,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "latency_distribution.html"
    fig.write_html(out_path, include_plotlyjs="inline", full_html=True)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
