"""Pydantic request/response schemas for the /detect-fraud endpoint.

Real-time fraud scoring needs the same velocity/geo/amount-deviation
features used in training, but those are defined relative to a customer's
transaction history -- a full historical join per request would blow the
latency budget. Production systems solve this with an online feature store
(e.g. Feast, or a Redis-backed aggregator) that keeps each customer's last
known state (last location, last timestamp, rolling mean/std of amount,
rolling counts) updated as a side effect of each transaction, and hands it to
the scoring service as O(1) state. ``CustomerState`` models exactly that
contract: the caller (the feature store) supplies the customer's last known
state, and this service only pays for the O(1) feature arithmetic + model
inference, which is what keeps it under the 50ms budget.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class CustomerState(BaseModel):
    """Lightweight rolling state a feature store would maintain per customer."""

    last_latitude: float | None = Field(None, description="Customer's last known transaction latitude")
    last_longitude: float | None = Field(None, description="Customer's last known transaction longitude")
    last_timestamp: str | None = Field(None, description="ISO-8601 timestamp of the customer's last transaction")
    home_latitude: float | None = Field(None, description="Customer's historical home-base latitude (expanding centroid)")
    home_longitude: float | None = Field(None, description="Customer's historical home-base longitude (expanding centroid)")
    avg_amount_clp: float | None = Field(None, description="Customer's historical average transaction amount (CLP)")
    std_amount_clp: float | None = Field(None, description="Customer's historical transaction amount std-dev (CLP)")
    avg_seconds_between_txn: float | None = Field(None, description="Customer's historical average gap between transactions, in seconds")
    txn_count_last_1h: int = Field(0, description="Number of transactions by this customer in the trailing 1 hour")
    txn_count_last_24h: int = Field(0, description="Number of transactions by this customer in the trailing 24 hours")
    amount_sum_last_1h: float = Field(0.0, description="CLP volume by this customer in the trailing 1 hour")


class TransactionRequest(BaseModel):
    transaction_id: str
    customer_id: int
    timestamp: str = Field(..., description="ISO-8601 timestamp of this transaction")
    amount_clp: float = Field(..., gt=0)
    merchant_category: str
    latitude: float
    longitude: float
    customer_state: CustomerState = Field(default_factory=CustomerState)


class FraudDetectionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    decision_threshold: float
    model_latency_ms: float
    reasons: list[str]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    feature_columns: list[str]
