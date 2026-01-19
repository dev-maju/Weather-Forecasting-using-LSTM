import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime
import joblib

# ===============================
# Base directory (MUST be first)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===============================
# Paths
# ===============================
MODEL_PATH = os.path.join(
    BASE_DIR, "..", "models", "kochi_lstm_next_hour_multivariate.keras"
)

DATA_PATH = os.path.join(
    BASE_DIR, "..", "data", "processed", "kochi_weather_scaled.csv"
)

SCALER_PATH = os.path.join(
    BASE_DIR, "..", "models", "weather_scaler.pkl"
)

LOG_PATH = os.path.join(
    BASE_DIR, "..", "results", "prediction_log.csv"
)

import joblib

SCALER_PATH = os.path.join(
    BASE_DIR, "..", "models", "weather_scaler.pkl"
)

scaler = joblib.load(SCALER_PATH)


MODEL_PATH = "../models/kochi_lstm_next_hour_multivariate.keras"
DATA_PATH = "../data/processed/kochi_weather_scaled.csv"
LOG_PATH = "../results/prediction_log.csv"
model = load_model(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
df['DATE'] = pd.to_datetime(df['DATE'])

TIME_STEPS = 24

data = df[['temperature', 'humidity', 'pressure']].values

latest_window = data[-TIME_STEPS:]
latest_window = latest_window.reshape(1, TIME_STEPS, 3)

prediction = model.predict(latest_window)

pred_scaled = prediction.reshape(1, -1)
pred_real = scaler.inverse_transform(pred_scaled)

pred_temp, pred_hum, pred_pres = pred_real[0]


print("Next-hour forecast:")
print(f"Temperature: {pred_temp:.4f}")
print(f"Humidity: {pred_hum:.4f}")
print(f"Pressure: {pred_pres:.4f}")

log_entry = {
    "timestamp": datetime.utcnow(),
    "pred_temperature": pred_temp,
    "pred_humidity": pred_hum,
    "pred_pressure": pred_pres
}

log_df = pd.DataFrame([log_entry])


log_df.to_csv(
    LOG_PATH,
    mode='a',
    header=not os.path.exists(LOG_PATH),
    index=False
)
