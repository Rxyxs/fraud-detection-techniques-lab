"""Geospatial features: distance and implied travel speed between consecutive
transactions of the same customer -- a classic fraud signal (an "impossible
travel" jump, e.g. Santiago -> Antofagasta in 3 minutes, cannot be a genuine
cardholder).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0

# A commercial flight cruises around 850-900 km/h; anything faster than this
# between two consecutive purchases is physically impossible for one person.
IMPOSSIBLE_TRAVEL_KMH = 900.0


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized great-circle distance in kilometers."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds, per customer (grouped, ordered by timestamp):

    - ``prev_latitude`` / ``prev_longitude``: location of the prior transaction.
    - ``distance_from_prev_km``: haversine distance from the prior transaction.
    - ``seconds_since_prev``: time gap to the prior transaction (also produced
      by ``time_velocity.add_time_velocity_features``; kept here too so this
      module is independently usable/testable).
    - ``implied_speed_kmh``: distance / time -- the "teleportation" signal.
    - ``is_impossible_travel``: 1 if implied speed exceeds a physically
      plausible bound for a single traveler.
    - ``distance_from_home_km``: distance from the customer's historical
      centroid (expanding mean of lat/lon up to and including this row).

    Assumes rows are already sorted by (customer_id, timestamp), as produced
    by ``generate_transactions.generate_dataset``.
    """
    df = df.copy()
    grp = df.groupby("customer_id", sort=False)

    prev_lat = grp["latitude"].shift(1)
    prev_lon = grp["longitude"].shift(1)
    df["prev_latitude"] = prev_lat
    df["prev_longitude"] = prev_lon

    has_prev = prev_lat.notna()
    dist = np.zeros(len(df))
    dist[has_prev.to_numpy()] = haversine_km(
        df.loc[has_prev, "latitude"].to_numpy(),
        df.loc[has_prev, "longitude"].to_numpy(),
        prev_lat[has_prev].to_numpy(),
        prev_lon[has_prev].to_numpy(),
    )
    df["distance_from_prev_km"] = dist

    seconds_since_prev = grp["timestamp"].diff().dt.total_seconds()
    df["seconds_since_prev"] = seconds_since_prev.fillna(np.inf)

    hours = df["seconds_since_prev"].to_numpy() / 3600.0
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = df["distance_from_prev_km"].to_numpy() / hours
    # hours == 0 with distance > 0 (two purchases at the identical timestamp in
    # different places) is a maximally strong impossible-travel signal, not a
    # missing value -> map that +inf to a large sentinel rather than to 0.
    speed = np.nan_to_num(speed, nan=0.0, posinf=IMPOSSIBLE_TRAVEL_KMH * 10, neginf=0.0)
    # First transaction per customer has no prior point -> speed undefined, set 0.
    speed[~has_prev.to_numpy()] = 0.0
    df["implied_speed_kmh"] = speed
    df["is_impossible_travel"] = (df["implied_speed_kmh"] > IMPOSSIBLE_TRAVEL_KMH).astype(int)

    # Expanding centroid = customer's "home base" estimated online (no lookahead).
    expanding_lat_mean = grp["latitude"].apply(lambda s: s.shift(1).expanding().mean())
    expanding_lon_mean = grp["longitude"].apply(lambda s: s.shift(1).expanding().mean())
    expanding_lat_mean = expanding_lat_mean.reset_index(level=0, drop=True)
    expanding_lon_mean = expanding_lon_mean.reset_index(level=0, drop=True)
    home_lat = expanding_lat_mean.fillna(df["latitude"])
    home_lon = expanding_lon_mean.fillna(df["longitude"])
    df["distance_from_home_km"] = haversine_km(
        df["latitude"].to_numpy(), df["longitude"].to_numpy(),
        home_lat.to_numpy(), home_lon.to_numpy(),
    )

    return df
