from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta

app = FastAPI()

# =============================
# LOAD MODEL + ENCODERS
# =============================
MODEL_DIR = "onion_models"

model = joblib.load(f"{MODEL_DIR}/onion_model.pkl")

encoders = {
    "Market": joblib.load(f"{MODEL_DIR}/Market_encoder.pkl"),
    "Grade": joblib.load(f"{MODEL_DIR}/Grade_encoder.pkl"),
    "Festival": joblib.load(f"{MODEL_DIR}/Festival_encoder.pkl"),
    "Tithi": joblib.load(f"{MODEL_DIR}/Tithi_encoder.pkl"),
    "Nakshatra": joblib.load(f"{MODEL_DIR}/Nakshatra_encoder.pkl"),
    "Heat Stress": joblib.load(f"{MODEL_DIR}/Heat Stress_encoder.pkl"),
    "Rain Alert": joblib.load(f"{MODEL_DIR}/Rain Alert_encoder.pkl"),
}

# =============================
# LOAD DATA FOR LAG
# =============================
df = pd.read_csv("nashik_onion_filtered.csv")
df["Date"] = pd.to_datetime(df["Date"])

# =============================
# COORDINATES
# =============================
coords = {
    "LASALGAON": (20.1420, 74.2390),
    "NANDGAON": (20.3000, 74.6500),
    "PIMPALGAON": (20.2000, 74.0000),
    "CHANDVAD": (20.3300, 74.2500),
    "MANMAD": (20.2500, 74.4500),
    "SINNER": (19.8500, 74.0000),
    "LASALGAON(NIPHAD)": (20.1420, 74.2390),
    "PIMPALGAON BASWANT(SAYKHEDA)": (20.1300, 73.9000),
    "LASALGAON(VINCHUR)": (20.1500, 74.2000),
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
        print(f"⚠️ Unknown {name}: {val}")
        val = encoder.classes_[0]
    return encoder.transform([val])[0]

# =============================
# FESTIVAL API
# =============================
def get_festival(date):
    try:
        url = f"https://calender-api-production.up.railway.app/calendar?date={date}"
        res = requests.get(url, timeout=5)

        if res.status_code != 200:
            raise Exception("Bad response")

        data = res.json()

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

    except Exception as e:
        print("❌ Festival API error:", e)
        return ("None", 0, 0, 0, "Krishna Paksha", "Ashwini")

# =============================
# WEATHER (1 CALL OPTIMIZED 🔥)
# =============================
def get_weather_forecast(lat, lon):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,precipitation_sum",
        "timezone": "auto"
    }

    try:
        res = requests.get(url, params=params, timeout=10)

        if res.status_code != 200:
            print("❌ Weather API error:", res.text)
            return {}

        data = res.json()

        forecast = {}

        for i, d in enumerate(data["daily"]["time"]):
            temp = data["daily"]["temperature_2m_max"][i]
            rain = data["daily"]["precipitation_sum"][i]

            heat = "Yes" if temp > 35 else "No"

            if rain > 20:
                rain_alert = "Heavy"
            elif rain > 5:
                rain_alert = "Moderate"
            else:
                rain_alert = "No"

            forecast[d] = (temp, rain, heat, rain_alert)

        return forecast

    except Exception as e:
        print("❌ Weather error:", e)
        return {}

# =============================
# LAG FEATURES
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
# API
# =============================
@app.get("/predict")
def predict(market: str, grade: str):

    market = market.upper()
    grade = grade.upper()

    start_date = datetime.today() + timedelta(days=1)

    lat, lon = coords[market]

    weather_data = get_weather_forecast(lat, lon)

    results = []

    for i in range(7):

        d = start_date + timedelta(days=i)
        date_str = d.strftime("%Y-%m-%d")

        # TIME
        month = d.month
        weekday = d.weekday()

        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        weekday_sin = np.sin(2 * np.pi * weekday / 7)
        weekday_cos = np.cos(2 * np.pi * weekday / 7)

        # FESTIVAL
        festival, ama, eka, pur, tithi, nak = get_festival(date_str)

        # WEATHER
        if date_str in weather_data:
            temp, rain, heat, rain_alert = weather_data[date_str]
        else:
            temp, rain, heat, rain_alert = 30, 0, "No", "No"

        # LAG
        lag = get_lag(market, grade)

        # FEATURES
        features = [[
            enc("Market", market),
            enc("Grade", grade),

            month_sin, month_cos,
            weekday_sin, weekday_cos,

            enc("Festival", festival),
            ama, eka, pur,
            enc("Tithi", tithi),
            enc("Nakshatra", nak),

            temp, rain,
            enc("Heat Stress", heat),
            enc("Rain Alert", rain_alert),

            lag["lag_1"],
            lag["lag_3"],
            lag["lag_7"],

            lag["rolling_mean_3"],
            lag["rolling_mean_7"],
            lag["rolling_std_7"],
            lag["price_diff_1"]
        ]]

        pred = model.predict(features)[0]

        results.append({
            "Date": date_str,
            "Market": market,
            "Grade": grade,

            "Min Price": int(pred[0]),
            "Max Price": int(pred[1]),
            "Modal Price": int(pred[2]),

            "Festival": festival,
            "is_amavasya": ama,
            "is_ekadashi": eka,
            "is_purnima": pur,
            "Tithi": tithi,
            "Nakshatra": nak,

            "Temp Max (°C)": temp,
            "Rain (mm)": rain,
            "Heat Stress": heat,
            "Rain Alert": rain_alert
        })

    return results