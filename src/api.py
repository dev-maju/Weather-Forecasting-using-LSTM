from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("firebase_key.json")

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://test1-a7529-default-rtdb.firebaseio.com/"
})

# -----------------------------
# App initialization
# -----------------------------
app = FastAPI(title="Weather Forecast API")

# -----------------------------
# BASE DIRECTORY
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# Model & scaler paths
# -----------------------------
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "kochi_lstm_next_hour_multivariate.keras"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "weather_scaler.pkl"
)

# -----------------------------
# Load artifacts
# -----------------------------
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

TIME_STEPS = 24
NUM_FEATURES = 3

# Dummy history for first 23 hours
weather_buffer = [[30,70,1005]] * 23

# -----------------------------
# Request schema (ESP32 input)
# -----------------------------
class WeatherReading(BaseModel):
    temperature: float
    humidity: float
    pressure: float

# -----------------------------
# Sensor endpoint
# -----------------------------
@app.post("/sensor")
def receive_sensor(data: WeatherReading):

    global weather_buffer

    reading = [
        data.temperature,
        data.humidity,
        data.pressure
    ]

    weather_buffer.append(reading)

    print("Received:", reading)
    print("Buffer size:", len(weather_buffer))

    # Wait until we have 24 readings
    if len(weather_buffer) < TIME_STEPS:
        return {"status": "collecting data"}

    # Use last 24 readings
    window = np.array(weather_buffer[-TIME_STEPS:], dtype=np.float32)

    # scale input
    window_scaled = scaler.transform(window)

    window_scaled = np.expand_dims(window_scaled, axis=0)

    prediction_scaled = model.predict(window_scaled)[0]

    dummy = np.zeros((1, NUM_FEATURES))
    dummy[0] = prediction_scaled

    prediction = scaler.inverse_transform(dummy)[0]

    prediction_data = {
    "temperature": float(prediction[0]),
    "humidity": float(prediction[1]),
    "pressure": float(prediction[2])
    }

    db.reference("ESP32/prediction").set(prediction_data)

    result = {
        "pred_temperature": float(prediction[0]),
        "pred_humidity": float(prediction[1]),
        "pred_pressure": float(prediction[2])
    }


    print("Prediction:", result)

    return result