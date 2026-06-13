"""
Synthetic data generator for Challenge 1.2 - Predictive Analytics for Resource
Allocation in Disaster & NGO Operations.

Focused use case: FLOOD-RELIEF KIT DEMAND FORECASTING — Coastal Odisha districts.

NOTE ON DATA SOURCE
--------------------
This environment has no live internet access to data.gov.in, IMD, NDMA or
Bhuvan/ISRO portals. To make the pipeline fully runnable and reproducible,
this script generates a SYNTHETIC-BUT-REALISTIC daily dataset whose
structure, scales and seasonal/spatial patterns mirror real Odisha
monsoon/cyclone dynamics (June-Sep monsoon peak, Oct-Nov cyclone season,
district-level population density from Census 2011 ballpark figures,
NDMA-style incident triggers based on river-gauge danger levels).

The code is written so that swapping in real CSVs from data.gov.in / IMD /
NDMA / SEDAC requires only changing the data-loading cell (see README) —
the feature engineering, modelling, dashboard and playbook all operate on
the same schema produced here.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. District master data (coastal Odisha — flood/cyclone prone)
#    Population density (people/km2) and centroids are realistic ballpark
#    figures (Census 2011 + standard district centroid coordinates).
# ---------------------------------------------------------------------------
DISTRICTS = {
    "Puri":           dict(lat=19.80, lon=85.83, pop_density=488, road_access=0.55, coast_km=5,  danger_gauge=5.5),
    "Ganjam":         dict(lat=19.40, lon=84.80, pop_density=611, road_access=0.60, coast_km=10, danger_gauge=6.0),
    "Kendrapara":     dict(lat=20.50, lon=86.60, pop_density=689, road_access=0.40, coast_km=8,  danger_gauge=5.0),
    "Balasore":       dict(lat=21.50, lon=86.93, pop_density=601, road_access=0.62, coast_km=15, danger_gauge=5.8),
    "Jagatsinghpur":  dict(lat=20.25, lon=86.17, pop_density=689, road_access=0.45, coast_km=6,  danger_gauge=5.2),
    "Bhadrak":        dict(lat=21.06, lon=86.50, pop_density=689, road_access=0.50, coast_km=20, danger_gauge=5.5),
    "Khordha":        dict(lat=20.18, lon=85.60, pop_density=854, road_access=0.75, coast_km=40, danger_gauge=4.5),
}

START = "2021-01-01"
END = "2024-12-31"  # 4 years of daily history for training/validation

dates = pd.date_range(START, END, freq="D")


def seasonal_rainfall(date, district_jitter):
    """Synthetic daily rainfall (mm) with SW monsoon (Jun-Sep) and
    post-monsoon cyclone season (Oct-Nov) peaks, Odisha-style."""
    doy = date.dayofyear
    # Monsoon bump centred on day ~200 (mid-July), cyclone bump ~ day 290 (mid-Oct)
    monsoon = 18 * np.exp(-((doy - 200) ** 2) / (2 * 35 ** 2))
    cyclone = 10 * np.exp(-((doy - 290) ** 2) / (2 * 15 ** 2))
    base = 1.0 + monsoon + cyclone
    # Occasional extreme-rain / cyclone-landfall days (heavy tail)
    extreme = 0.0
    if np.random.rand() < (0.015 if 150 <= doy <= 320 else 0.002):
        extreme = np.random.gamma(shape=2.0, scale=40) * district_jitter
    noise = max(0, np.random.gamma(shape=1.2, scale=base))
    return round(noise + extreme, 1)


# ---------------------------------------------------------------------------
# 2. Build long daily panel: district x date
# ---------------------------------------------------------------------------
records = []
for dist, meta in DISTRICTS.items():
    jitter = np.random.uniform(0.8, 1.3)
    gauge_level = meta["danger_gauge"] * 0.6  # starting baseline gauge level
    incident_log = []  # rolling list of incident flags for last 30 days

    for date in dates:
        rain = seasonal_rainfall(date, jitter)

        # River gauge level: persists + responds to rainfall (accumulation/decay)
        gauge_level = (
            0.92 * gauge_level
            + 0.02 * rain
            + np.random.normal(0, 0.03)
        )
        gauge_level = max(0.5, gauge_level)

        # Incident trigger: gauge above danger mark -> "flood incident day"
        incident = int(gauge_level >= meta["danger_gauge"])
        incident_log.append(incident)
        if len(incident_log) > 30:
            incident_log.pop(0)
        past_incident_30d = sum(incident_log)

        records.append(
            dict(
                date=date,
                district=dist,
                rainfall_mm=rain,
                river_gauge_m=round(gauge_level, 2),
                population_density=meta["pop_density"],
                road_accessibility_index=meta["road_access"],
                distance_to_coast_km=meta["coast_km"],
                danger_gauge_m=meta["danger_gauge"],
                past_incident_30d=past_incident_30d,
                lat=meta["lat"],
                lon=meta["lon"],
            )
        )

df = pd.DataFrame.from_records(records)

# ---------------------------------------------------------------------------
# 3. Target variable: relief_kits_demand
#    A "relief kit" bundles: 1 tarpaulin + dry-ration pack + ORS sachets +
#    10L drinking water, sized per SPHERE-style per-capita norms.
#    Demand scales with: rainfall severity, gauge exceedance above danger
#    level, population density (affected population proxy), road
#    inaccessibility (harder access -> more pre-positioning needed),
#    and recent incident frequency (compounding vulnerability).
# ---------------------------------------------------------------------------
df["gauge_exceed"] = (df["river_gauge_m"] - df["danger_gauge_m"]).clip(lower=0)
df["rain_severity"] = (df["rainfall_mm"] / 50).clip(upper=4)  # normalised severity

affected_pop_proxy = (df["population_density"] / 600) * (1 + df["gauge_exceed"])
demand_base = (
    affected_pop_proxy
    * (1 + 1.8 * df["rain_severity"])
    * (1 + 0.5 * (1 - df["road_accessibility_index"]))
    * (1 + 0.15 * df["past_incident_30d"])
)
noise = np.random.normal(1.0, 0.08, size=len(df)).clip(0.7, 1.3)
df["relief_kits_demand"] = np.maximum(0, (demand_base * 60 * noise)).round().astype(int)

# Final column order
cols = [
    "date", "district", "rainfall_mm", "river_gauge_m", "danger_gauge_m",
    "gauge_exceed", "population_density", "road_accessibility_index",
    "distance_to_coast_km", "past_incident_30d", "lat", "lon",
    "relief_kits_demand",
]
df = df[cols]

out_path = "/home/claude/project/data/odisha_flood_relief_synthetic.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df):,} rows to {out_path}")
print(df.groupby('district')['relief_kits_demand'].describe()[['mean','std','max']])
