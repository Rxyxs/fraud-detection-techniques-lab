"""Validates the < 50ms inference-latency requirement for /detect-fraud.

Two latencies are measured separately:

- ``model_latency_ms`` (from the response body): feature computation +
  autoencoder + CatBoost + XGBoost inference, timed inside the endpoint.
  This is the number the 50ms budget is actually about.
- End-to-end wall-clock time through the ASGI TestClient (request
  validation, routing, JSON serialization included), as a sanity check that
  framework overhead alone doesn't blow the budget.
"""
import statistics
import time

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from tests.test_api import LEGIT_PAYLOAD

N_REQUESTS = 200
LATENCY_BUDGET_MS = 50.0


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _percentile(values, pct):
    return sorted(values)[int(pct * (len(values) - 1))]


def test_model_latency_p95_under_budget(client):
    latencies = []
    for i in range(N_REQUESTS):
        payload = {**LEGIT_PAYLOAD, "transaction_id": f"TXN_LAT_{i}"}
        r = client.post("/detect-fraud", json=payload)
        assert r.status_code == 200
        latencies.append(r.json()["model_latency_ms"])

    p50 = statistics.median(latencies)
    p95 = _percentile(latencies, 0.95)
    print(f"model_latency_ms: p50={p50:.2f}ms p95={p95:.2f}ms max={max(latencies):.2f}ms")

    assert p95 < LATENCY_BUDGET_MS


def test_end_to_end_roundtrip_p95_under_budget(client):
    latencies_ms = []
    for i in range(N_REQUESTS):
        payload = {**LEGIT_PAYLOAD, "transaction_id": f"TXN_E2E_{i}"}
        start = time.perf_counter()
        r = client.post("/detect-fraud", json=payload)
        elapsed = (time.perf_counter() - start) * 1000.0
        assert r.status_code == 200
        latencies_ms.append(elapsed)

    p95 = _percentile(latencies_ms, 0.95)
    print(f"end_to_end_ms: p95={p95:.2f}ms max={max(latencies_ms):.2f}ms")
    assert p95 < LATENCY_BUDGET_MS
