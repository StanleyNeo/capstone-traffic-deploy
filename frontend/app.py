"""Streamlit frontend for the Smart City Traffic capstone (Render deployment).

Two tabs:
  - Predict:   calls the FastAPI backend /predict endpoint
  - Visuals:   shows the 4 Part 2 exploratory figures

Backend URL is read from the BACKEND_URL environment variable, so the same
image works locally (default http://127.0.0.1:8000) and on Render.
"""
import os
from pathlib import Path

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
FIG_DIR = Path(__file__).resolve().parent / "figures"

WEATHER_CONDITIONS = ["Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist",
                      "Rain", "Smoke", "Snow", "Squall", "Thunderstorm"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title="Smart City Traffic — Capstone",
                   page_icon="🚦", layout="wide")

st.title("🚦 Smart City Traffic — Metro Interstate")
st.caption("Part 2 (visuals) + Part 3 (ML predictions) served live from Cloudflare Tunnel.")

# ---------- backend health ----------
with st.sidebar:
    st.header("Backend")
    st.code(BACKEND_URL, language="text")
    try:
        h = requests.get(f"{BACKEND_URL}/health", timeout=10).json()
        if h.get("models_loaded"):
            st.success("Backend online — models loaded ✅")
        else:
            st.warning("Backend up but models NOT loaded")
    except Exception as e:
        st.error(f"Backend unreachable:\n{e}")

tab_predict, tab_visuals = st.tabs(["🎯 Predict", "📊 Visuals"])

# ---------- Predict tab ----------
with tab_predict:
    st.subheader("Predict traffic volume & congestion risk")
    st.write("Enter one hour of conditions. The backend engineers all 27 features.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            hour = st.slider("Hour (0-23)", 0, 23, 17)
            dow_name = st.selectbox("Day of week", DAYS, index=2)
            dow = DAYS.index(dow_name)
        with c2:
            month_name = st.selectbox("Month", MONTHS, index=5)
            month = MONTHS.index(month_name) + 1
            is_holiday = st.checkbox("Public holiday", value=False)
        with c3:
            temp_c = st.number_input("Temperature (°C)", -30.0, 40.0, 25.0, 0.5)
            weather_main = st.selectbox("Weather", WEATHER_CONDITIONS, index=0)

        c4, c5, c6 = st.columns(3)
        with c4:
            rain_1h = st.number_input("Rain last hour (mm)", 0.0, 100.0, 0.0, 0.1)
        with c5:
            snow_1h = st.number_input("Snow last hour (mm)", 0.0, 100.0, 0.0, 0.1)
        with c6:
            clouds_all = st.slider("Cloud cover (%)", 0, 100, 50)

        submitted = st.form_submit_button("Predict", width="stretch")

    if submitted:
        payload = {
            "hour": hour, "day_of_week": dow, "month": month,
            "temp_c": temp_c, "rain_1h": rain_1h, "snow_1h": snow_1h,
            "clouds_all": clouds_all, "weather_main": weather_main,
            "is_holiday": is_holiday,
        }
        try:
            r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=20)
            r.raise_for_status()
            out = r.json()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted volume", f"{out['predicted_traffic_volume']:.0f} veh/h")
            m2.metric("Congestion prob.", f"{out['congestion_probability']:.2%}")
            m3.metric("Congested?", "YES" if out["congestion_predicted"] else "no")
            m4.metric("Risk level", out["risk_level"])
            if out["risk_level"] == "HIGH":
                st.error("🔴 HIGH congestion risk")
            elif out["risk_level"] == "MODERATE":
                st.warning("🟠 MODERATE congestion risk")
            else:
                st.success("🟢 LOW congestion risk")
            with st.expander("Raw JSON response"):
                st.json(out)
        except Exception as e:
            st.error(f"Request failed: {e}")

# ---------- Visuals tab ----------
with tab_visuals:
    st.subheader("Part 2 — Exploratory analysis")
    st.caption("Generated from the cleaned 40,575-row dataset (2012-2018).")
    figures = [
        ("fig1_hourly_profile.png",
         "Fig 1 — Hourly traffic profile (weekday peak 16:00, weekend peak 13:00)"),
        ("fig2_heatmap.png",
         "Fig 2 — Day-of-week × hour heatmap (busiest: Wed 16:00)"),
        ("fig3_weather_impact.png",
         "Fig 3 — Traffic by weather condition (highest: Clouds)"),
        ("fig4_monthly_rhythm.png",
         "Fig 4 — Monthly rhythm (highest: Aug, lowest: Dec)"),
    ]
    for fname, caption in figures:
        p = FIG_DIR / fname
        if p.exists():
            st.image(str(p), caption=caption, width="stretch")
        else:
            st.warning(f"Missing figure: {fname}")

