# 🌊 Flood-Relief Resource Allocation Forecast — Coastal Odisha

**Challenge 1.2 — Predictive Analytics for Resource Allocation in Disaster & NGO Operations**
*(Track: AI & Intelligent Systems)*

A machine-learning system that forecasts **relief-kit demand (drinking water,
dry rations, ORS, tarpaulin, medical kits)** over the next **7–30 days** for
7 flood/cyclone-prone coastal Odisha districts — Puri, Ganjam, Kendrapara,
Balasore, Jagatsinghpur, Bhadrak, and Khordha — so NGOs and district disaster
cells can pre-position supplies *before* a flood event, instead of reacting
after.

---

## 📁 Project Structure
flood_relief_project/

├── README.md                  ← you are here

├── requirements.txt

├── dashboard/

│   └── app.py                 # Streamlit decision-support dashboard

├── src/

│   ├── generate_data.py       # synthetic-but-realistic data generator

│   ├── train_model.py          # feature engineering + LightGBM training

│   └── forecast.py             # forward forecast + supply breakdown

├── data/

│   └── odisha_flood_relief_synthetic.csv   # 4-year daily dataset, 7 districts

└── outputs/

├── flood_relief_forecasting.ipynb      # full reproducible pipeline (executed)

├── resource_allocation_playbook.docx   # 1-page NGO ops playbook

├── district_risk_map.html              # interactive folium map

├── accuracy_report.json                # MAE / RMSE / MAPE per horizon

├── forecast_results.csv                # latest forecast + supply breakdown

├── model_point_h{7,14,30}.txt          # trained LightGBM point models

├── model_q10_h*.txt / model_q90_h*.txt # 10th/90th percentile uncertainty models
└── feature_importance.csv, holdout_predictions.csv, featured_dataset.csv,

latest_features.csv, feature_list.json   # supporting artefacts
---

---

## 🎯 Results

| Horizon | MAE (kits) | RMSE (kits) | MAPE | Target |
|---|---|---|---|---|
| 7 days  | ~354  | ~1,304 | **~6.2%** | < 20% ✅ |
| 14 days | ~525  | ~1,819 | **~4.0%** | < 20% ✅ |
| 30 days | ~1,207 | ~3,916 | **~3.4%** | < 20% ✅ |

Top predictors: **rainfall** (current + 1/3/7/14-day lags), **river-gauge
exceedance above the danger mark**, and **recent demand momentum** —
mirroring the real rainfall → river-rise → incident → relief-need chain.

---

## 🚀 Quick Start

### 1. Clone and set up
```bash
git clone https://github.com/kumar-chandan23116027/flood-relief-forecast.git
cd flood-relief-forecast
python -m pip install -r requirements.txt
```

### 2. Run the dashboard
```bash
cd dashboard
python -m streamlit run app.py
```
Opens at `http://localhost:8501` with:
- **District map** — ranked relief-kit demand by district, colour-coded
- **Scenario simulator** — plug in an IMD rainfall outlook / CWC gauge
  reading and get a live forecast + supply breakdown with uncertainty bands
- **Model performance** — holdout accuracy, feature importance
- **Playbook** — decision tiers and operating rhythm

### 3. Explore the notebook
Open `outputs/flood_relief_forecasting.ipynb` in Jupyter — it's fully
executed and reproducible end-to-end (data generation → EDA → features →
training → evaluation → forecast → map).

### 4. Retrain from scratch
```bash
python src/generate_data.py     # writes data/odisha_flood_relief_synthetic.csv
python src/train_model.py        # writes models + accuracy_report.json to outputs/
python src/forecast.py           # writes forecast_results.csv + latest_features.csv
```

---

## 🧠 How It Works

1. **Data** — daily panel per district: rainfall, river-gauge level, danger
   threshold, population density, road-accessibility index, distance to
   coast, 30-day incident count.
2. **Features** — rainfall/gauge lags (1/3/7/14 days), rolling sums/means
   (3/7/14/30 days), seasonality (day-of-year sin/cos, monsoon/cyclone-season
   flags), gauge trend.
3. **Models** — LightGBM regressors, one per horizon (7/14/30 days):
   - a **point estimate** model (MAE objective)
   - **10th/90th percentile quantile models** for uncertainty bands
4. **Output** — total relief-kit demand per district per horizon, broken
   down into individual supplies via SPHERE-standard per-capita ratios.

---

## ⚠️ Data Note — Going Live with Real Data

The included dataset is **synthetic but structurally realistic** (built to
mirror IMD rainfall seasonality, CWC gauge dynamics, Census-2011 population
density, NDMA-style incident triggers), since this was developed without live
API access to data.gov.in / IMD / NDMA / Bhuvan / SEDAC.

To go live, replace `src/generate_data.py`'s output with real data pulled
from those sources, **keeping the same column schema**:
Then re-run `train_model.py` and `forecast.py` unchanged. Retrain monthly
during monsoon season (Jun–Nov) and quarterly otherwise, feeding back logged
actual kit-distribution numbers as new ground truth.

---

## 📋 Operational Playbook (summary)

| Tier | Trigger | Action |
|---|---|---|
| 🟢 Watch | Gauge below danger mark, baseline forecast | Confirm warehouse stock; no movement |
| 🟡 Alert | Forecast rainfall > 50mm/day or rising gauge trend | Move **lower-bound (10th pct)** quantity to staging point within 48h |
| 🔴 Action | Gauge ≥ danger mark or cyclone warning | Dispatch **upper-bound (90th pct)** quantity to flood shelters; activate field teams |

Full details, kit composition, and uncertainty-communication guidance are in
`outputs/resource_allocation_playbook.docx`.

---

## 📚 Reference

SPHERE Handbook (Humanitarian Charter and Minimum Standards) for per-capita
supply norms. Live-deployment data sources: data.gov.in, NDMA incident
database, IMD, Bhuvan (ISRO), OpenStreetMap, SEDAC population grids.
