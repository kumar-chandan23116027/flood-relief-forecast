"""
Feature engineering + LightGBM model training for flood-relief kit demand
forecasting (Challenge 1.2).

Target definition
------------------
For each (district, date) we predict TOTAL relief-kit demand over the next
H days (H is configurable: 7, 14, or 30 — the "7-30 day" horizon called for
in the problem statement). We train one model per horizon.

Outputs
-------
- /home/claude/project/outputs/model_h{H}.txt   (LightGBM models, point + quantiles)
- /home/claude/project/outputs/accuracy_report.json
- /home/claude/project/outputs/feature_importance.csv
- /home/claude/project/outputs/holdout_predictions.csv
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "/home/claude/project/data/odisha_flood_relief_synthetic.csv"
OUT_DIR = "/home/claude/project/outputs"
HORIZONS = [7, 14, 30]

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values(["district", "date"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
gb = df.groupby("district")
for lag in [1, 3, 7, 14]:
    df[f"rainfall_lag{lag}"] = gb["rainfall_mm"].shift(lag)
    df[f"gauge_lag{lag}"] = gb["river_gauge_m"].shift(lag)
for win in [3, 7, 14, 30]:
    df[f"rainfall_roll{win}"] = gb["rainfall_mm"].transform(lambda s: s.rolling(win).sum())
    df[f"demand_roll{win}"] = gb["relief_kits_demand"].transform(lambda s: s.rolling(win).mean())
df["gauge_trend_7"] = df["river_gauge_m"] - gb["river_gauge_m"].shift(7)

# Seasonality features
df["doy"] = df["date"].dt.dayofyear
df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365.25)
df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365.25)
df["is_monsoon"] = df["date"].dt.month.isin([6, 7, 8, 9]).astype(int)
df["is_cyclone_season"] = df["date"].dt.month.isin([10, 11]).astype(int)

# District as categorical
df["district_cat"] = df["district"].astype("category")

BASE_FEATURES = [
    "rainfall_mm", "river_gauge_m", "gauge_exceed", "danger_gauge_m",
    "population_density", "road_accessibility_index", "distance_to_coast_km",
    "past_incident_30d",
    "rainfall_lag1", "rainfall_lag3", "rainfall_lag7", "rainfall_lag14",
    "gauge_lag1", "gauge_lag3", "gauge_lag7", "gauge_lag14",
    "rainfall_roll3", "rainfall_roll7", "rainfall_roll14", "rainfall_roll30",
    "demand_roll3", "demand_roll7", "demand_roll14", "demand_roll30",
    "gauge_trend_7",
    "sin_doy", "cos_doy", "is_monsoon", "is_cyclone_season",
    "district_cat",
]

accuracy_report = {}
feature_importances = []
holdout_frames = []

for H in HORIZONS:
    work = df.copy()
    # Target: sum of demand over the NEXT H days (rolling forward sum, shifted)
    work[f"target_h{H}"] = (
        work.groupby("district")["relief_kits_demand"]
        .transform(lambda s: s.shift(-1).rolling(H).sum())
    )
    work = work.dropna(subset=[f"target_h{H}"] + BASE_FEATURES[:-1]).reset_index(drop=True)

    # Time-based split: last 6 months as holdout
    cutoff = work["date"].max() - pd.Timedelta(days=180)
    train = work[work["date"] <= cutoff]
    test = work[work["date"] > cutoff]

    X_train, y_train = train[BASE_FEATURES], train[f"target_h{H}"]
    X_test, y_test = test[BASE_FEATURES], test[f"target_h{H}"]

    cat_features = ["district_cat"]

    # --- Point estimate model (regression, MAE objective for robustness) ---
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train, categorical_feature=cat_features)
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)

    # --- Quantile models for uncertainty band (10th / 90th percentile) ---
    q_models = {}
    q_preds = {}
    for q in [0.1, 0.9]:
        qm = lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, random_state=42, verbosity=-1,
        )
        qm.fit(X_train, y_train, categorical_feature=cat_features)
        q_preds[q] = np.clip(qm.predict(X_test), 0, None)
        q_models[q] = qm

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mape = np.mean(np.abs((y_test - preds) / np.maximum(y_test, 1))) * 100

    accuracy_report[f"horizon_{H}d"] = {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "MAPE_pct": round(float(mape), 2),
        "n_test": int(len(test)),
        "n_train": int(len(train)),
    }

    # Save models
    model.booster_.save_model(f"{OUT_DIR}/model_point_h{H}.txt")
    q_models[0.1].booster_.save_model(f"{OUT_DIR}/model_q10_h{H}.txt")
    q_models[0.9].booster_.save_model(f"{OUT_DIR}/model_q90_h{H}.txt")

    # Feature importance
    fi = pd.DataFrame({
        "feature": BASE_FEATURES,
        "importance": model.feature_importances_,
        "horizon": H,
    }).sort_values("importance", ascending=False)
    feature_importances.append(fi)

    # Holdout predictions for plotting
    hold = test[["date", "district", f"target_h{H}"]].copy()
    hold["prediction"] = preds
    hold["q10"] = q_preds[0.1]
    hold["q90"] = q_preds[0.9]
    hold["horizon"] = H
    hold = hold.rename(columns={f"target_h{H}": "actual"})
    holdout_frames.append(hold)

    print(f"Horizon {H}d -> MAE={mae:.1f}, RMSE={rmse:.1f}, MAPE={mape:.1f}%")

# Save artifacts
with open(f"{OUT_DIR}/accuracy_report.json", "w") as f:
    json.dump(accuracy_report, f, indent=2)

pd.concat(feature_importances).to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)
pd.concat(holdout_frames).to_csv(f"{OUT_DIR}/holdout_predictions.csv", index=False)

# Save the feature list + dataframe needed for live forecasting in the dashboard
df.to_csv(f"{OUT_DIR}/featured_dataset.csv", index=False)
with open(f"{OUT_DIR}/feature_list.json", "w") as f:
    json.dump(BASE_FEATURES, f)

print("\nSaved models, accuracy report, feature importances, holdout predictions.")
