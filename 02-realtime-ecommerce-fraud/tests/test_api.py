import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


LEGIT_PAYLOAD = {
    "transaction_id": "TXN_TEST_LEGIT",
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


def _fraud_payload():
    payload = {**LEGIT_PAYLOAD, "transaction_id": "TXN_TEST_FRAUD", "amount_clp": 950000}
    payload["latitude"] = -20.21
    payload["longitude"] = -70.15
    payload["customer_state"] = {
        **LEGIT_PAYLOAD["customer_state"],
        "last_timestamp": "2026-03-15T14:29:00",
        "txn_count_last_1h": 4,
    }
    return payload


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "amount_clp" in body["feature_columns"]


def test_detect_fraud_legit_like_transaction(client):
    r = client.post("/detect-fraud", json=LEGIT_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["transaction_id"] == "TXN_TEST_LEGIT"
    assert body["is_fraud"] is False
    assert 0.0 <= body["fraud_probability"] < 0.5


def test_detect_fraud_flags_extreme_burst_and_geo_jump(client):
    r = client.post("/detect-fraud", json=_fraud_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["is_fraud"] is True
    assert body["fraud_probability"] > 0.5
    assert len(body["reasons"]) >= 1
    assert body["reasons"][0] != "No individual high-risk signal crossed its alert threshold."


def test_detect_fraud_rejects_nonpositive_amount(client):
    bad_payload = {**LEGIT_PAYLOAD, "amount_clp": -100}
    r = client.post("/detect-fraud", json=bad_payload)
    assert r.status_code == 422


def test_detect_fraud_handles_missing_customer_state(client):
    # A brand-new customer with no history at all -> customer_state defaults.
    payload = {**LEGIT_PAYLOAD, "transaction_id": "TXN_NEW_CUSTOMER"}
    payload.pop("customer_state")
    r = client.post("/detect-fraud", json=payload)
    assert r.status_code == 200
