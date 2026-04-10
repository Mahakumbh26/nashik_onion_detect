import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="Onion Price Predictor", layout="wide")

# =============================
# LOAD MODEL (FIXED)
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "onion_models")

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(MODEL_DIR, "onion_model.pkl"))

@st.cache_resource
def load_encoders():
    return {
        "Market": joblib.load(os.path.join(MODEL_DIR, "Market_encoder.pkl")),
        "Grade": joblib.load(os.path.join(MODEL_DIR, "Grade_encoder.pkl")),
        "Festival": joblib.load(os.path.join(MODEL_DIR, "Festival_encoder.pkl")),
        "Tithi": joblib.load(os.path.join(MODEL_DIR, "Tithi_encoder.pkl")),
        "Nakshatra": joblib.load(os.path.join(MODEL_DIR, "Nakshatra_encoder.pkl")),
        "Heat Stress": joblib.load(os.path.join(MODEL_DIR, "Heat Stress_encoder.pkl")),
        "Rain Alert": joblib.load(os.path.join(MODEL_DIR, "Rain Alert_encoder.pkl")),
    }

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "nashik_onion_filtered.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    return df

model = load_model()
encoders = load_encoders()
df = load_data()

# =============================
# UI
# =============================
st.title("🧅 Onion Price Forecast Dashboard")

MARKETS = [
    "LASALGAON","NANDGAON","PIMPALGAON","CHANDVAD",
    "MANMAD","SINNER","YEOLA","UMRANE","KALVAN"
]

GRADES = ["OTHER","POLE","RED","WHITE"]

col1, col2 = st.columns(2)
market = col1.selectbox("Select Market", MARKETS)
grade = col2.selectbox("Select Grade", GRADES)

# =============================
# COORDS
# =============================
coords = {
    "LASALGAON": (20.1420, 74.2390),
    "NANDGAON": (20.3000, 74.6500),
    "PIMPALGAON": (20.2000, 74.0000),
    "CHANDVAD": (20.3300, 74.2500),
    "MANMAD": (20.2500, 74.4500),
    "SINNER": (19.8500, 74.0000),
    "YEOLA": (20.0500, 74.5000),
    "UMRANE": (20.3000, 74.2000),
    "KALVAN": (20.5000, 74.0000),
}

# =============================
# SAFE ENCODER
# =============================
def enc(name, val):
    encoder = encoders[name]
    if val not in encoder.classes_:
        val = encoder.classes_[0]
    return encoder.transform([val])[0]

# =============================
# FESTIVAL
# =============================
def get_festival(date):
    try:
        url = f"https://calender-api-production.up.railway.app/calendar?date={date}"
        data = requests.get(url, timeout=5).json()

        fest_list = data.get("state_festivals", {}).get("Maharashtra", [])
        festival = ", ".join(fest_list) if fest_list else "None"

        return (
            festival,
            int(data.get("is_amavasya", False)),
            int(data.get("is_ekadashi", False)),
            int(data.get("is_purnima", False)),
            data.get("tithi", "Krishna Paksha"),
            data.get("nakshatra", "Ashwini")
        )
    except:
        return ("None", 0, 0, 0, "Krishna Paksha", "Ashwini")

# =============================
# WEATHER (1 CALL)
# =============================
def get_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,precipitation_sum",
        "timezone": "auto"
    }
    try:
        data = requests.get(url, params=params, timeout=5).json()
        weather = {}

        for i, d in enumerate(data["daily"]["time"]):
            temp = data["daily"]["temperature_2m_max"][i]
            rain = data["daily"]["precipitation_sum"][i]

            heat = "Yes" if temp > 35 else "No"

            if rain > 20:
                ra = "Heavy"
            elif rain > 5:
                ra = "Moderate"
            else:
                ra = "No"

            weather[d] = (temp, rain, heat, ra)

        return weather
    except:
        return {}

# =============================
# LAG
# =============================
def get_lag(market, grade):
    subset = df[(df["Market"] == market) & (df["Grade"] == grade)].sort_values("Date")
    last = subset.tail(7)
    p = last["Modal Price (Rs./Qtl)"].values

    return {
        "lag_1": p[-1],
        "lag_3": p[-3],
        "lag_7": p[0],
        "rolling_mean_3": p[-3:].mean(),
        "rolling_mean_7": p.mean(),
        "rolling_std_7": p.std(),
        "price_diff_1": p[-1] - p[-2]
    }

# =============================
# PREDICT BUTTON
# =============================
if st.button("🔮 Predict 7 Days"):

    lat, lon = coords[market]
    weather = get_weather(lat, lon)

    start_date = datetime.today() + timedelta(days=1)

    results = []

    for i in range(7):

        d = start_date + timedelta(days=i)
        ds = d.strftime("%Y-%m-%d")

        month = d.month
        weekday = d.weekday()

        festival, ama, eka, pur, tithi, nak = get_festival(ds)

        temp, rain, heat, ra = weather.get(ds, (30, 0, "No", "No"))

        lag = get_lag(market, grade)

        features = [[
            enc("Market", market),
            enc("Grade", grade),
            np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12),
            np.sin(2*np.pi*weekday/7), np.cos(2*np.pi*weekday/7),
            enc("Festival", festival),
            ama, eka, pur,
            enc("Tithi", tithi),
            enc("Nakshatra", nak),
            temp, rain,
            enc("Heat Stress", heat),
            enc("Rain Alert", ra),
            lag["lag_1"], lag["lag_3"], lag["lag_7"],
            lag["rolling_mean_3"], lag["rolling_mean_7"],
            lag["rolling_std_7"], lag["price_diff_1"]
        ]]

        pred = model.predict(features)[0]

        results.append({
            "Date": ds,
            "Min Price": int(pred[0]),
            "Max Price": int(pred[1]),
            "Modal Price": int(pred[2]),
            "Festival": festival,
            "Tithi": tithi,
            "Nakshatra": nak,
            "Temp": temp,
            "Rain": rain,
            "Heat Stress": heat,
            "Rain Alert": ra
        })

    df_res = pd.DataFrame(results)

    # =============================
    # INTERACTIVE GRAPH
    # =============================
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_res["Date"], y=df_res["Min Price"],
        mode='lines+markers', name='Min Price'
    ))

    fig.add_trace(go.Scatter(
        x=df_res["Date"], y=df_res["Modal Price"],
        mode='lines+markers', name='Modal Price'
    ))

    fig.add_trace(go.Scatter(
        x=df_res["Date"], y=df_res["Max Price"],
        mode='lines+markers', name='Max Price'
    ))

    fig.update_layout(
        title="📈 7-Day Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price (Rs/Qtl)",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =============================
    # TABLE
    # =============================
    st.subheader("📋 Detailed Forecast Data")
    st.dataframe(df_res, use_container_width=True)