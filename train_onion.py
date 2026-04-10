import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# =============================
# CONFIG
# =============================
DATA_FILE = "nashik_onion_filtered.csv"
MODEL_DIR = "onion_models"

os.makedirs(MODEL_DIR, exist_ok=True)

# =============================
# LOAD DATA
# =============================
df = pd.read_csv(DATA_FILE)

# =============================
# CLEAN
# =============================
df.columns = df.columns.str.strip()

df["Market"] = df["Market"].astype(str).str.upper().str.strip()
df["Grade"] = df["Grade"].astype(str).str.upper().str.strip()

df["Date"] = pd.to_datetime(df["Date"])

# =============================
# SORT (VERY IMPORTANT)
# =============================
df = df.sort_values(["Market", "Grade", "Date"])

# =============================
# TIME FEATURES (ENCODED)
# =============================
df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

df["weekday"] = df["Date"].dt.weekday
df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

# =============================
# LAG FEATURES 🔥
# =============================
df["lag_1"] = df.groupby(["Market","Grade"])["Modal Price (Rs./Qtl)"].shift(1)
df["lag_3"] = df.groupby(["Market","Grade"])["Modal Price (Rs./Qtl)"].shift(3)
df["lag_7"] = df.groupby(["Market","Grade"])["Modal Price (Rs./Qtl)"].shift(7)

# =============================
# ROLLING FEATURES
# =============================
df["rolling_mean_3"] = df.groupby(["Market","Grade"])["Modal Price (Rs./Qtl)"].transform(lambda x: x.rolling(3).mean())
df["rolling_mean_7"] = df.groupby(["Market","Grade"])["Modal Price (Rs./Qtl)"].transform(lambda x: x.rolling(7).mean())
df["rolling_std_7"] = df.groupby(["Market","Grade"])["Modal Price (Rs./Qtl)"].transform(lambda x: x.rolling(7).std())

# =============================
# MOMENTUM
# =============================
df["price_diff_1"] = df.groupby(["Market","Grade"])["Modal Price (Rs./Qtl)"].diff(1)

# =============================
# FILL CATEGORICAL
# =============================
df["Festival"] = df["Festival"].fillna("None")
df["Tithi"] = df["Tithi"].fillna("Unknown")
df["Nakshatra"] = df["Nakshatra"].fillna("Unknown")
df["Heat Stress"] = df["Heat Stress"].fillna("No")
df["Rain Alert"] = df["Rain Alert"].fillna("No")

# =============================
# ENCODING
# =============================
encoders = {}

def encode(col):
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

encode("Market")
encode("Grade")
encode("Festival")
encode("Tithi")
encode("Nakshatra")
encode("Heat Stress")
encode("Rain Alert")

# =============================
# FEATURES
# =============================
features = [
    "Market_enc",
    "Grade_enc",

    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",

    "Festival_enc",
    "is_amavasya",
    "is_ekadashi",
    "is_purnima",
    "Tithi_enc",
    "Nakshatra_enc",

    "Temp Max (°C)",
    "Rain (mm)",
    "Heat Stress_enc",
    "Rain Alert_enc",

    "lag_1",
    "lag_3",
    "lag_7",

    "rolling_mean_3",
    "rolling_mean_7",
    "rolling_std_7",
    "price_diff_1"
]

# =============================
# TARGET
# =============================
target = [
    "Min Price (Rs./ Qtl)",
    "Max Price (Rs./Qtl)",
    "Modal Price (Rs./Qtl)"
]

# =============================
# DROP NA (VERY IMPORTANT)
# =============================
df = df.dropna(subset=features + target)

X = df[features]
y = df[target]

print("📊 Training samples:", len(X))

# =============================
# MODEL
# =============================
model = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
)

# =============================
# TRAIN
# =============================
print("🚀 Training Onion Model...")
model.fit(X, y)

# =============================
# SAVE MODEL + ENCODERS
# =============================
joblib.dump(model, f"{MODEL_DIR}/onion_model.pkl")

for name, enc in encoders.items():
    joblib.dump(enc, f"{MODEL_DIR}/{name}_encoder.pkl")

print("\n🔥 MODEL TRAINED & SAVED SUCCESSFULLY!")
print(f"📁 Saved in folder: {MODEL_DIR}/")