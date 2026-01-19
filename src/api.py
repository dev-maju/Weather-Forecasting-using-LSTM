from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# -----------------------------
# App initialization
# -----------------------------
app = FastAPI(title="Weather Forecast API")

# -----------------------------
# BASE DIRECTORY (MUST BE FIRST)
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

# -----------------------------
# Request schema
# -----------------------------
class WeatherRequest(BaseModel):
    weather_window: list[list[float]]

# -----------------------------
# Response schema
# -----------------------------
class WeatherResponse(BaseModel):
    temperature: float
    humidity: float
    pressure: float

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict", response_model=WeatherResponse)
def predict(request: WeatherRequest):

    data = np.array(request.weather_window, dtype=np.float32)

    if data.shape != (TIME_STEPS, NUM_FEATURES):
        raise HTTPException(
            status_code=400,
            detail="Input must be of shape (24, 3)"
        )

    data = np.expand_dims(data, axis=0)

    prediction_scaled = model.predict(data)[0]

    dummy = np.zeros((1, NUM_FEATURES))
    dummy[0] = prediction_scaled
    prediction = scaler.inverse_transform(dummy)[0]

    return {
        "temperature": float(prediction[0]),
        "humidity": float(prediction[1]),
        "pressure": float(prediction[2]),
    }
