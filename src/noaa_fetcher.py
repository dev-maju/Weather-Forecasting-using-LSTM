"""

NOAA real-time ingestion module

NOTE:
Disabled due to upstream NOAA data latency.
Pipeline supports activation when data becomes available.


import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# ---------------- CONFIG ----------------
STATION_ID = "43353099999"  # Kochi Airport
BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
DATASET = "global-hourly"

PARAMS = [
    "DATE",
    "TMP",
    "DEW",
    "SLP"
]

RAW_DATA_PATH = "../data/raw/noaa_latest.csv"

# ---------------- FETCH FUNCTION ----------------
def fetch_latest_noaa(days_back=5):
    """
    Fetch latest available NOAA data.
    Falls back gracefully if NOAA is delayed.
    """

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days_back)

    params = {
        "dataset": DATASET,
        "stations": STATION_ID,
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "dataTypes": ",".join(PARAMS),
        "format": "csv",
        "units": "metric"
    }

    print(f"[INFO] Fetching NOAA data from {start_date} to {end_date}")

    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(pd.compat.StringIO(response.text))

        if df.empty:
            print("[WARNING] NOAA returned empty data. Using last saved file if available.")
            return load_fallback()

        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"[SUCCESS] NOAA data fetched ({len(df)} rows)")
        return df

    except Exception as e:
        print(f"[ERROR] NOAA fetch failed: {e}")
        return load_fallback()

# ---------------- FALLBACK ----------------
def load_fallback():
    if os.path.exists(RAW_DATA_PATH):
        print("[INFO] Loading fallback NOAA data")
        return pd.read_csv(RAW_DATA_PATH)

    print("[CRITICAL] No NOAA data available at all.")
    return pd.DataFrame()

# ---------------- RUN DIRECTLY ----------------
if __name__ == "__main__":
    df = fetch_latest_noaa()
    print(df.tail())
"""