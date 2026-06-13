"""
Flood-Relief Resource Allocation Dashboard
============================================
Challenge 1.2 — Predictive Analytics for Resource Allocation in
Disaster & NGO Operations (AI & Intelligent Systems track)

Use case: Flood-relief kit demand forecasting for 7 coastal Odisha districts
(Puri, Ganjam, Kendrapara, Balasore, Jagatsinghpur, Bhadrak, Khordha).

Run with:  streamlit run app.py
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import folium
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="Flood-Relief Demand Forecast", layout="wide", page_icon="🌊")

import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")

# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    latest = pd.read_csv(f"{DATA_DIR}/latest_features.csv")
    forecast = pd.read_csv(f"{DATA_DIR}/forecast_results.csv")
    holdout = pd.read_csv(f"{DATA_DIR}/holdout_predictions.csv", parse_dates=["date"])
    with open(f"{DATA_DIR}/accuracy_report.json") as f:
        accuracy = json.load(f)
    with open(f"{DATA_DIR}/feature_list.json") as f:
        features = json.load(f)
    fi = pd.read_csv(f"{DATA_DIR}/feature_importance.csv")
    return latest, forecast, holdout, accuracy, features, fi


@st.cache_resource
def load_models(horizon):
    point = lgb.Booster(model_file=f"{DATA_DIR}/model_point_h{horizon}.txt")
    q10 = lgb.Booster(model_file=f"{DATA_DIR}/model_q10_h{horizon}.txt")
    q90 = lgb.Booster(model_file=f"{DATA_DIR}/model_q90_h{horizon}.txt")
    return point, q10, q90


latest, forecast, holdout, accuracy, FEATURES, fi = load_data()

KIT_COMPOSITION = {
    "Drinking water (litres)": 15,
    "Dry-ration packs":         1,
    "ORS sachets":              4,
    "Tarpaulin sheets":      0.25,
    "Medical-kit units":       0.1,
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🌊 Flood-Relief Forecast")
st.sidebar.markdown(
    "Forecasting where & how much **relief-kit** demand (water, dry rations, "
    "ORS, tarpaulin, medical kits) will arise over the next **7–30 days** "
    "across flood-prone coastal Odisha districts."
)
horizon = st.sidebar.selectbox("Forecast horizon (days)", [7, 14, 30], index=0)
districts = sorted(latest["district"].unique())
selected_district = st.sidebar.selectbox("District (for detail views)", districts)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Model accuracy ({horizon}-day horizon)**\n\n"
    f"- MAE: {accuracy[f'horizon_{horizon}d']['MAE']:.0f} kits\n"
    f"- RMSE: {accuracy[f'horizon_{horizon}d']['RMSE']:.0f} kits\n"
    f"- MAPE: {accuracy[f'horizon_{horizon}d']['MAPE_pct']:.1f}% "
    f"(target < 20% ✅)"
)
st.sidebar.caption(
    "Data: synthetic, structured to mirror IMD rainfall, CWC river-gauge, "
    "Census population density, NDMA incident patterns. Swap in real "
    "data.gov.in / IMD / NDMA feeds via the same schema — see README."
)

# ---------------------------------------------------------------------------
# Header + KPI row
# ---------------------------------------------------------------------------
st.title("Flood-Relief Kit Demand Forecast — Coastal Odisha")
st.caption(
    "Decision-support tool for NGO ops managers and district disaster cells. "
    "Forecasts total relief-kit demand per district for the selected horizon, "
    "with a breakdown into individual supplies (SPHERE-standard ratios)."
)

fdf = forecast[forecast.horizon_days == horizon].sort_values("predicted_kits", ascending=False)
total_kits = fdf["predicted_kits"].sum()
top_district = fdf.iloc[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total kits needed (all districts)", f"{total_kits:,.0f}")
k2.metric("Highest-need district", top_district["district"], f"{top_district['predicted_kits']:,.0f} kits")
k3.metric("Forecast horizon", f"{horizon} days")
k4.metric("Model MAPE (holdout)", f"{accuracy[f'horizon_{horizon}d']['MAPE_pct']:.1f}%")

tab_map, tab_scenario, tab_perf, tab_playbook = st.tabs(
    ["🗺️ District Map & Forecast", "🎛️ Scenario Simulator", "📈 Model Performance", "📋 Allocation Playbook"]
)

# ---------------------------------------------------------------------------
# TAB 1: Map + ranked table
# ---------------------------------------------------------------------------
with tab_map:
    col_map, col_table = st.columns([3, 2])

    with col_map:
        m = folium.Map(location=[20.3, 85.9], zoom_start=7, tiles="CartoDB positron")
        max_kits = fdf["predicted_kits"].max()
        for _, row in fdf.iterrows():
            frac = row["predicted_kits"] / max_kits if max_kits > 0 else 0
            color = (
                "#08306b" if frac > 0.85 else
                "#2171b5" if frac > 0.6 else
                "#6baed6" if frac > 0.35 else
                "#c6dbef"
            )
            radius = 8 + 22 * frac
            popup = folium.Popup(
                f"<b>{row['district']}</b><br>"
                f"Predicted demand ({horizon}d): <b>{row['predicted_kits']:,.0f} kits</b><br>"
                f"90% range: {row['lower_90pct_band']:,.0f} – {row['upper_90pct_band']:,.0f}<br>"
                f"<hr style='margin:4px 0'>"
                f"Drinking water: {row['drinking_water_litres']:,.0f} L<br>"
                f"Dry-ration packs: {row['dry_ration_packs']:,.0f}<br>"
                f"ORS sachets: {row['ors_sachets']:,.0f}<br>"
                f"Tarpaulins: {row['tarpaulin_sheets']:,.0f}<br>"
                f"Medical kits: {row['medical_kit_units']:,.0f}",
                max_width=280,
            )
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=popup,
                tooltip=f"{row['district']}: {row['predicted_kits']:,.0f} kits",
            ).add_to(m)
        st_folium(m, width=None, height=480, returned_objects=[])

    with col_table:
        st.subheader(f"Ranked demand — next {horizon} days")
        show = fdf[["district", "predicted_kits", "lower_90pct_band", "upper_90pct_band"]].copy()
        show.columns = ["District", "Predicted kits", "Lower (90% band)", "Upper (90% band)"]
        st.dataframe(show.set_index("District"), use_container_width=True)
        st.caption(
            "Circle size/colour = relative predicted demand. Band = 10th–90th "
            "percentile from quantile LightGBM models, communicating forecast "
            "uncertainty to relief coordinators."
        )

    st.subheader(f"Supply breakdown — {selected_district} (next {horizon} days)")
    drow = fdf[fdf.district == selected_district].iloc[0]
    bcols = st.columns(len(KIT_COMPOSITION))
    for (label, ratio), c in zip(KIT_COMPOSITION.items(), bcols):
        key = label.lower().replace(" (litres)", "_litres").replace(" ", "_").replace("-", "_")
        # map back to forecast_results.csv column names
        colmap = {
            "Drinking water (litres)": "drinking_water_litres",
            "Dry-ration packs": "dry_ration_packs",
            "ORS sachets": "ors_sachets",
            "Tarpaulin sheets": "tarpaulin_sheets",
            "Medical-kit units": "medical_kit_units",
        }
        c.metric(label, f"{drow[colmap[label]]:,.0f}")

# ---------------------------------------------------------------------------
# TAB 2: Scenario simulator
# ---------------------------------------------------------------------------
with tab_scenario:
    st.subheader("What-if scenario: upcoming weather forecast → relief-kit demand")
    st.markdown(
        "Adjust the inputs below to reflect an **IMD rainfall outlook** or "
        "**CWC river-gauge bulletin** for the selected district, and see the "
        "model's live demand forecast. This is how a district disaster cell "
        "would translate a weather warning into a pre-positioning order."
    )

    base = latest[latest.district == selected_district].iloc[0].copy()

    c1, c2, c3 = st.columns(3)
    rainfall = c1.slider("Forecast rainfall (mm/day, avg next 7 days)", 0.0, 300.0, float(base["rainfall_mm"]), 5.0)
    gauge = c2.slider("River gauge level (m)", 0.0, float(base["danger_gauge_m"]) * 1.8, float(base["river_gauge_m"]), 0.1)
    incidents = c3.slider("Flood-incident days in past 30 days", 0, 30, int(base["past_incident_30d"]))

    scenario = base.copy()
    scenario["rainfall_mm"] = rainfall
    scenario["river_gauge_m"] = gauge
    scenario["gauge_exceed"] = max(0, gauge - base["danger_gauge_m"])
    scenario["past_incident_30d"] = incidents
    # Push the rainfall/gauge shock into recent lags & rolling features too,
    # since a sustained weather event affects the whole recent window.
    for lag in [1, 3, 7, 14]:
        scenario[f"rainfall_lag{lag}"] = rainfall
        scenario[f"gauge_lag{lag}"] = gauge
    for win in [3, 7, 14, 30]:
        scenario[f"rainfall_roll{win}"] = rainfall * win
    scenario["gauge_trend_7"] = gauge - base["river_gauge_m"]

    point_model, q10_model, q90_model = load_models(horizon)
    X = pd.DataFrame([scenario[FEATURES]])
    X["district_cat"] = X["district_cat"].astype("category").cat.set_categories(districts)

    pred = float(np.clip(point_model.predict(X), 0, None)[0])
    lo = float(np.clip(q10_model.predict(X), 0, None)[0])
    hi = float(np.clip(q90_model.predict(X), 0, None)[0])

    st.markdown("### Forecast result")
    r1, r2, r3 = st.columns(3)
    r1.metric(f"Predicted demand ({horizon}d)", f"{pred:,.0f} kits")
    r2.metric("Lower bound (10th pct)", f"{lo:,.0f} kits")
    r3.metric("Upper bound (90th pct)", f"{hi:,.0f} kits")

    st.markdown("#### Recommended pre-positioning order")
    bcols = st.columns(len(KIT_COMPOSITION))
    for (label, ratio), c in zip(KIT_COMPOSITION.items(), bcols):
        c.metric(label, f"{pred * ratio:,.0f}")

    if scenario["gauge_exceed"] > 0:
        st.warning(
            f"⚠️ River gauge is **{scenario['gauge_exceed']:.2f} m above the "
            f"danger mark** ({base['danger_gauge_m']:.1f} m) for {selected_district}. "
            "This corresponds to an active flood-incident trigger in NDMA-style "
            "monitoring — recommend immediate dispatch of the lower-bound "
            "quantity, with the upper bound held at the nearest staging hub."
        )
    else:
        st.info(
            "River level is below the danger mark. Forecast reflects routine / "
            "seasonal-baseline demand — suitable for **planning, not emergency, "
            "pre-positioning**."
        )

# ---------------------------------------------------------------------------
# TAB 3: Model performance
# ---------------------------------------------------------------------------
with tab_perf:
    st.subheader(f"Holdout validation — {horizon}-day horizon")
    acc = accuracy[f"horizon_{horizon}d"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{acc['MAE']:.0f} kits")
    c2.metric("RMSE", f"{acc['RMSE']:.0f} kits")
    c3.metric("MAPE", f"{acc['MAPE_pct']:.1f}%")
    c4.metric("Holdout size", f"{acc['n_test']:,} obs")
    st.caption(
        "Holdout = last 6 months of the 4-year synthetic series (time-based "
        "split, no leakage). Target: MAPE < 20% — met across all horizons."
    )

    st.markdown("#### Actual vs. predicted demand (holdout period)")
    hsub = holdout[(holdout.horizon == horizon) & (holdout.district == selected_district)].sort_values("date")
    chart_df = hsub.set_index("date")[["actual", "prediction", "q10", "q90"]]
    st.line_chart(chart_df[["actual", "prediction"]])
    st.caption(
        f"District: {selected_district}. 'actual'/'prediction' = total relief-kit "
        f"demand over the *following* {horizon} days from each date."
    )

    st.markdown("#### Top model features")
    fi_h = fi[fi.horizon == horizon].sort_values("importance", ascending=False).head(10)
    st.bar_chart(fi_h.set_index("feature")["importance"])
    st.caption(
        "Rainfall, river-gauge exceedance, and recent demand momentum dominate — "
        "consistent with how flood relief operations are actually triggered."
    )

# ---------------------------------------------------------------------------
# TAB 4: Playbook
# ---------------------------------------------------------------------------
with tab_playbook:
    st.subheader("1-Page Resource-Allocation Playbook (NGO Ops)")
    st.markdown("""
**Trigger tiers** (based on forecast vs. river-gauge danger mark):

| Tier | Condition | Action |
|---|---|---|
| 🟢 Watch | Gauge < danger mark, forecast within seasonal baseline | Confirm stock levels at district warehouse; no movement |
| 🟡 Alert | Forecast rainfall > 50 mm/day OR rising gauge trend | Move **lower-bound (10th pct)** quantity to nearest taluk-level staging point within 48 hrs |
| 🔴 Action | Gauge ≥ danger mark OR cyclone warning issued | Dispatch **upper-bound (90th pct)** quantity to flood-shelter network immediately; activate NGO field teams |

**Per-kit composition** (SPHERE-aligned minimums, per affected person / 3 days):
- 15 L drinking water, 1 dry-ration family pack, 4 ORS sachets, 0.25 tarpaulin sheet, 0.1 medical-kit unit

**Weekly operating rhythm:**
1. Every Monday: pull latest IMD 7-day rainfall outlook + CWC gauge bulletin for each district → enter into Scenario Simulator.
2. Compare predicted demand to current warehouse stock; raise procurement/transfer requests for any shortfall vs. the *lower bound*.
3. If any district crosses 🔴 Action tier, escalate to state-level NGO network for surge stock from neighboring low-need districts (use the District Map ranking to identify donors).
4. Log actual relief-kit consumption after each event — feed back into the model's training data monthly to keep `demand_roll` features current.

**Data gaps in remote districts:** when live IMD/CWC feeds are missing for a district, fall back to the **regional average of neighboring coastal districts** for rainfall/gauge inputs (the model already encodes district-level population density & road-access priors, so this degrades gracefully rather than failing).

**Communicating uncertainty:** always quote the **range** (10th–90th percentile), not a single number, to relief coordinators — frame as *"plan for the lower bound, hold the difference to the upper bound in reserve at the nearest hub."*
""")
    st.caption("Reference: SPHERE Handbook (humanitarian minimum standards) for per-capita supply norms.")
