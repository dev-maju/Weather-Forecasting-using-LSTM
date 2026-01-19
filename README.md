# Weather Forecasting Using Multivariate LSTM (Kochi, India)
## Project Overview

This project implements an end-to-end weather forecasting system using a multivariate LSTM deep learning model to predict next-hour temperature, humidity, and atmospheric pressure. The system is designed with production-style architecture, covering the complete machine learning lifecycle — from data preprocessing and model training to API deployment.

The project was developed as a self-driven academic and practical exercise to build a strong foundation in:
  - Machine Learning
  - Deep Learning with Keras
  - Time-series forecasting
  - Feature engineering
  - Model deployment and MLOps concepts

## Problem Statement
Weather parameters exhibit temporal dependencies and non-linear relationships. Traditional models struggle to capture long-term patterns.
This project leverages Long Short-Term Memory (LSTM) networks to model these dependencies and provide accurate short-term forecasts.

## Dataset
Source: NOAA Global Hourly Weather Data
Location: Kochi, Kerala, India
Time Span: 2020 – 2024
Features Used:
  - Temperature
  - Relative Humidity
  - Sea Level Pressure
Real-time NOAA ingestion is intentionally disabled due to upstream data latency.
The pipeline is ingestion-ready and can be activated when live data becomes available.

## Data Preprocessing & Feature Engineering
Missing value handling
Unit normalization
Feature scaling using MinMaxScaler
Sliding window sequence creation (24-hour lookback)
Multivariate time-series formatting for LSTM input

## Model Architecture
Model Type: Multivariate LSTM
Framework: TensorFlow / Keras
Input Shape: (24, 3) → last 24 hours of weather data
Output: Next-hour prediction of:
  - Temperature
  - Humidity
  - Pressure
Loss Function: Mean Squared Error (MSE)
Optimizer: Adam
The trained model is saved in native Keras format (.keras) for production compatibility.

## Rolling Forecast Strategy
The system supports rolling predictions, where:
  - The latest observed data is continuously appended
  - Older data is discarded
  - The model predicts the next hour iteratively
This simulates real-time inference in production environments.

## API Deployment
A FastAPI-based REST API is implemented for inference.

### Endpoint
```python
POST /predict
```

### Input
JSON payload containing the last 24 hours of scaled weather data

### Output
Predicted next-hour:
  - Temperature
  - Humidity
  - Pressure
Interactive API documentation is available via Swagger UI.

## Project Structure
<pre>
weather-lstm-forecasting/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_lstm_training.ipynb
│
├── src/
│   ├── dataset_builder.py
│   ├── model.py
│   ├── rolling_predictor.py
│   ├── api.py
│   └── noaa_fetcher.py
│
├── models/
│   ├── kochi_lstm_next_hour_multivariate.keras
│   └── weather_scaler.pkl
│
├── results/
│   ├── plots/
│   └── metrics/
│
├── README.md
└── requirements.txt
</pre>

## Technologies Used
- Python
- NumPy, Pandas
- Scikit-learn
- TensorFlow / Keras
- FastAPI
- Joblib
- Matplotlib
- Uvicorn

## Key Learnings
- Time-series data preprocessing
- Multivariate LSTM modeling
- Feature scaling and inverse transformation
- Rolling window forecasting
- API-based ML model deployment
- Production-ready ML system design
- Practical MLOps concepts

## Future Improvements
- Enable live NOAA ingestion when data latency improves
- Add model performance monitoring
- Extend prediction horizon (multi-step forecasting)
- Containerize API using Docker
- Add automated retraining pipeline
