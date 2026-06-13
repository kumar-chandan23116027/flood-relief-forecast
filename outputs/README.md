# Challenge 1.2 — Predictive Analytics for Resource Allocation in Disaster & NGO Operations

**Track:** AI & Intelligent Systems
**Chosen use case:** Flood-relief kit demand forecasting for 7 flood/cyclone-prone
coastal Odisha districts — Puri, Ganjam, Kendrapara, Balasore, Jagatsinghpur,
Bhadrak, Khordha.

## What's in this package

| File | Purpose |
|---|---|
| `flood_relief_forecasting.ipynb` | **Reproducible notebook** — data generation, EDA, feature engineering, model training, accuracy report, forecast, map. Already executed end-to-end. |
| `resource_allocation_playbook.docx` | **1-page NGO playbook** — decision tiers, kit composition, weekly rhythm, handling data gaps, communicating uncertainty. |
| `district_risk_map.html` | Standalone interactive Folium map of 7-day relief-kit demand by district. |
| `accuracy_report.json` | MAE / RMSE / MAPE for the 7, 14, 30-day forecast horizons. |
| `forecast_results.csv` | Latest forecast + SPHERE-standard supply breakdown per district per horizon. |
| `feature_importance.csv`, `holdout_predictions.csv`, `featured_dataset.csv`, `feature_list.json`, `latest_features.csv` | Supporting model artefacts used by the dashboard. |
| `model_point_h{7,14,30}.txt`, `model_q10_h*.txt`, `model_q90_h*.txt` | Trained LightGBM models (point estimate + 10th/90th percentile quantile models) per horizon. |
| `dashboard/app.py` (separate folder) | **Streamlit dashboard** for NGO ops managers — map, scenario simulator, model performance, playbook. |

## Results

| Horizon | MAE (kits) | RMSE (kits) | MAPE | Target |
|---|---|---|---|---|
| 7 days  | ~354  | ~1,304 | **~6.2%** | < 20% ✅ |
| 14 days | ~525  | ~1,819 | **~4.0%** | < 20% ✅ |
| 30 days | ~1,207 | ~3,916 | **~3.4%** | < 20% ✅ |

Top predictive features across horizons: rainfall (current + 1/3/7/14-day lags),
river-gauge exceedance above the danger mark, and recent demand momentum
(7/14/30-day rolling means) — consistent with how flood-relief operations are
actually triggered in practice (rainfall → river rise → incident → relief need).

## Key questions addressed

- **Which features matter most?** Rainfall and river-gauge exceedance dominate,
  followed by recent-demand momentum and population density. Road accessibility
  and past-incident frequency contribute as secondary modifiers (see
  `feature_importance.csv`).
- **Handling data gaps in remote districts:** the playbook specifies a
  neighbour-average fallback for missing rainfall/gauge feeds; the model's
  reliance on static district priors (population density, road access, distance
  to coast) means it degrades gracefully rather than failing when a live feed
  is missing.
- **Communicating uncertainty:** every forecast ships with a 10th/90th
  percentile band from dedicated LightGBM quantile models. The dashboard and
  playbook frame this as "move the lower bound now, hold the rest in reserve" —
  giving a relief coordinator a clear yes/no action without hiding the
  uncertainty.

## Running the dashboard

```bash
cd dashboard
pip install streamlit streamlit-folium lightgbm folium pandas numpy
streamlit run app.py
```

The dashboard reads model artefacts and data from `../outputs/` (adjust the
`DATA_DIR` constant in `app.py` if you relocate the files — keep `outputs/`
alongside `dashboard/` for a drop-in run).

## Data note — and how to go live with real data

This sandbox has **no live access** to data.gov.in, IMD, NDMA, Bhuvan (ISRO) or
SEDAC APIs. The notebook's Section 1 therefore generates a
**synthetic-but-structurally-realistic** daily panel: rainfall follows the SW
monsoon (Jun–Sep) + Oct–Nov cyclone-season seasonality typical of Odisha, river
gauges respond to rainfall with realistic accumulation/decay, district
population densities are Census-2011 ballparks, and flood "incidents" are
triggered exactly as NDMA-style monitoring would (gauge ≥ danger mark).

To go live:
1. Replace Section 1 with real loaders — IMD (rainfall + forecast), CWC/NDMA
   (river-gauge levels, incident logs), Census/SEDAC (population density),
   Bhuvan/OpenStreetMap (road network → accessibility index) — **keeping the
   same column schema** (`date, district, rainfall_mm, river_gauge_m,
   danger_gauge_m, gauge_exceed, population_density, road_accessibility_index,
   distance_to_coast_km, past_incident_30d, lat, lon, relief_kits_demand`).
2. Re-run Sections 2–6 unchanged — feature engineering, training, evaluation,
   forecasting and mapping all operate on that schema.
3. Retrain monthly during monsoon season (Jun–Nov) and quarterly otherwise,
   using logged actual kit-distribution numbers as new ground truth for
   `relief_kits_demand`.

## Stretch goal — IMD forecast API integration

The `rainfall_mm` / `rainfall_lag*` / `rainfall_roll*` features are exactly the
slot where a real IMD extended-range / nowcast API would plug in: pull the
7-day rainfall outlook per district, feed it into the Scenario Simulator (or
directly into the feature pipeline) to get a forward-looking demand forecast
instead of the "persist current conditions" baseline used for the snapshot in
`forecast_results.csv`.
