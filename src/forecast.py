"""
Generate:
1. latest_features.csv  - most recent feature row per district (used as the
   "current state" seed for live forecasting / scenario simulation in the
   dashboard).
2. forecast_results.csv - point + uncertainty forecast (7/14/30-day total
   relief-kit demand) per district, plus a per-resource SPHERE-style
   breakdown (tarpaulins, dry-ration packs, ORS sachets, drinking-water
   litres, medical kits).
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb

OUT_DIR = "/home/claude/project/outputs"

df = pd.read_csv(f"{OUT_DIR}/featured_dataset.csv", parse_dates=["date"])
with open(f"{OUT_DIR}/feature_list.json") as f:
    FEATURES = json.load(f)

HORIZONS = [7, 14, 30]

# ---------------------------------------------------------------------------
# Latest feature snapshot per district (last row with complete features)
# ---------------------------------------------------------------------------
df_complete = df.dropna(subset=[c for c in FEATURES if c != "district_cat"])
latest = df_complete.sort_values("date").groupby("district").tail(1).reset_index(drop=True)
latest.to_csv(f"{OUT_DIR}/latest_features.csv", index=False)
print("Latest snapshot date per district:")
print(latest[["district", "date", "rainfall_mm", "river_gauge_m", "gauge_exceed"]])

# ---------------------------------------------------------------------------
# SPHERE-style per-kit composition (per "relief kit" = serves ~1 person for
# ~3 days, based on SPHERE Handbook minimum standards ballparks)
# ---------------------------------------------------------------------------
KIT_COMPOSITION = {
    "drinking_water_litres":  15,    # 15L / person / 3 days  (~5 L/day SPHERE min for drinking+cooking)
    "dry_ration_packs":        1,    # 1 family-ration pack
    "ors_sachets":              4,
    "tarpaulin_sheets":       0.25,  # 1 tarpaulin shared per ~4 kits/people
    "medical_kit_units":      0.1,   # 1 basic medical kit per ~10 kits
}

# ---------------------------------------------------------------------------
# Forecast: load saved models, predict for the latest snapshot
# ---------------------------------------------------------------------------
records = []
for H in HORIZONS:
    point_model = lgb.Booster(model_file=f"{OUT_DIR}/model_point_h{H}.txt")
    q10_model = lgb.Booster(model_file=f"{OUT_DIR}/model_q10_h{H}.txt")
    q90_model = lgb.Booster(model_file=f"{OUT_DIR}/model_q90_h{H}.txt")

    X = latest[FEATURES].copy()
    X["district_cat"] = X["district_cat"].astype("category")

    point = np.clip(point_model.predict(X), 0, None)
    q10 = np.clip(q10_model.predict(X), 0, None)
    q90 = np.clip(q90_model.predict(X), 0, None)

    for i, row in latest.iterrows():
        rec = dict(
            district=row["district"],
            horizon_days=H,
            predicted_kits=round(float(point[i]), 1),
            lower_90pct_band=round(float(q10[i]), 1),
            upper_90pct_band=round(float(q90[i]), 1),
            lat=row["lat"], lon=row["lon"],
        )
        for item, ratio in KIT_COMPOSITION.items():
            rec[item] = round(float(point[i]) * ratio, 1)
        records.append(rec)

forecast_df = pd.DataFrame(records)
forecast_df.to_csv(f"{OUT_DIR}/forecast_results.csv", index=False)
print("\nForecast (7-day horizon):")
print(forecast_df[forecast_df.horizon_days == 7][
    ["district", "predicted_kits", "lower_90pct_band", "upper_90pct_band"]
].to_string(index=False))
