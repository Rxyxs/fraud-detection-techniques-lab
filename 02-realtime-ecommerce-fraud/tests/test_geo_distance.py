import numpy as np
import pandas as pd

from src.features.geo_distance import IMPOSSIBLE_TRAVEL_KMH, add_geo_features, haversine_km


def test_haversine_known_distance():
    # Santiago -> Valparaiso is approximately 100-110 km in a straight line.
    santiago = (-33.4489, -70.6693)
    valparaiso = (-33.0472, -71.6127)
    dist = haversine_km(*santiago, *valparaiso)
    assert 90 < dist < 120


def test_haversine_zero_for_same_point():
    dist = haversine_km(-33.45, -70.66, -33.45, -70.66)
    assert abs(dist) < 1e-9


def _make_df():
    return pd.DataFrame({
        "customer_id": [1, 1, 1, 2, 2],
        "timestamp": pd.to_datetime([
            "2026-01-01 10:00:00",
            "2026-01-01 10:05:00",  # 5 min later, same city -> plausible
            "2026-01-01 10:06:00",  # 1 min later, but far away -> impossible travel
            "2026-01-01 09:00:00",
            "2026-01-01 09:30:00",
        ]),
        "amount_clp": [10000, 12000, 500, 5000, 5500],
        "latitude": [-33.45, -33.46, -18.47, -36.82, -36.83],
        "longitude": [-70.66, -70.65, -70.30, -73.04, -73.05],
    })


def test_add_geo_features_first_row_has_no_prior():
    df = add_geo_features(_make_df())
    first_rows = df[df["customer_id"] == 1].iloc[0]
    assert first_rows["distance_from_prev_km"] == 0.0
    assert first_rows["implied_speed_kmh"] == 0.0
    assert first_rows["is_impossible_travel"] == 0


def test_add_geo_features_flags_impossible_travel():
    df = add_geo_features(_make_df())
    cust1 = df[df["customer_id"] == 1].reset_index(drop=True)
    # Row 2: same city, 5 min gap, small distance -> not impossible travel.
    assert cust1.loc[1, "is_impossible_travel"] == 0
    # Row 3: ~1600km jump (central Chile -> northern Chile) in 1 minute -> impossible.
    assert cust1.loc[2, "distance_from_prev_km"] > 1000
    assert cust1.loc[2, "implied_speed_kmh"] > IMPOSSIBLE_TRAVEL_KMH
    assert cust1.loc[2, "is_impossible_travel"] == 1


def test_add_geo_features_independent_customers_dont_leak():
    df = add_geo_features(_make_df())
    cust2 = df[df["customer_id"] == 2].reset_index(drop=True)
    # Customer 2's first row must not see customer 1's prior location.
    assert cust2.loc[0, "distance_from_prev_km"] == 0.0
