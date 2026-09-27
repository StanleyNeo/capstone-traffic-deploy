"""FastAPI service for the Smart City Traffic models (Render deployment)."""
import logging, math, os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib, pandas as pd, uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

WEATHER_CONDITIONS = ["Clear","Clouds","Drizzle","Fog","Haze","Mist",
                      "Rain","Smoke","Snow","Squall","Thunderstorm"]
RUSH_HOURS = [6,7,8,9,15,16,17,18]
FEATURES = ["temp_c","rain_1h","snow_1h","clouds_all","hour","day_of_week","month",
            "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
            "is_weekend","is_rush_hour","is_holiday",
            "weather_Clear","weather_Clouds","weather_Drizzle","weather_Fog",
            "weather_Haze","weather_Mist","weather_Rain","weather_Smoke",
            "weather_Snow","weather_Squall","weather_Thunderstorm"]


class TrafficInput(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6, description="0=Mon ... 6=Sun")
    month: int = Field(..., ge=1, le=12)
    temp_c: float = Field(...)
    rain_1h: float = Field(0.0, ge=0.0)
    snow_1h: float = Field(0.0, ge=0.0)
    clouds_all: float = Field(50.0, ge=0.0, le=100.0)
    weather_main: str = Field("Clear")
    is_holiday: bool = Field(False)


class Prediction(BaseModel):
    predicted_traffic_volume: float
    congestion_probability: float
    congestion_predicted: bool
    risk_level: str


def build_feature_vector(inp: TrafficInput) -> pd.DataFrame:
    row = {
        "temp_c": inp.temp_c, "rain_1h": inp.rain_1h, "snow_1h": inp.snow_1h,
        "clouds_all": inp.clouds_all, "hour": inp.hour,
        "day_of_week": inp.day_of_week, "month": inp.month,
        "hour_sin": math.sin(2*math.pi*inp.hour/24),
        "hour_cos": math.cos(2*math.pi*inp.hour/24),
        "dow_sin":  math.sin(2*math.pi*inp.day_of_week/7),
        "dow_cos":  math.cos(2*math.pi*inp.day_of_week/7),
        "month_sin":math.sin(2*math.pi*inp.month/12),
        "month_cos":math.cos(2*math.pi*inp.month/12),
        "is_weekend": int(inp.day_of_week >= 5),
        "is_rush_hour": int(inp.hour in RUSH_HOURS),
        "is_holiday": int(inp.is_holiday),
    }
    for w in WEATHER_CONDITIONS:
        row[f"weather_{w}"] = int(inp.weather_main == w)
    return pd.DataFrame([row])[FEATURES]


regressor = None
classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global regressor, classifier
    reg_path = MODEL_DIR / "traffic_regressor.joblib"
    cls_path = MODEL_DIR / "congestion_classifier.joblib"
    if not reg_path.exists() or not cls_path.exists():
        logger.error("Model files missing in %s", MODEL_DIR)
        raise RuntimeError("Model files not found")
    regressor = joblib.load(reg_path)
    classifier = joblib.load(cls_path)
    logger.info("Models loaded from %s", MODEL_DIR)
    yield


app = FastAPI(title="Smart City Traffic API",
              description="Metro Interstate traffic volume + congestion risk",
              version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok",
            "models_loaded": regressor is not None and classifier is not None}


@app.post("/predict", response_model=Prediction)
def predict(inp: TrafficInput):
    if inp.weather_main not in WEATHER_CONDITIONS:
        raise HTTPException(422, detail=f"weather_main must be one of {WEATHER_CONDITIONS}")
    if regressor is None or classifier is None:
        raise HTTPException(503, detail="Models not loaded")
    X = build_feature_vector(inp)
    X = X[FEATURES]  # columns already in training order
    volume = float(regressor.predict(X)[0])
    proba = float(classifier.predict_proba(X)[0, 1])
    risk = "HIGH" if proba >= 0.75 else "MODERATE" if proba >= 0.4 else "LOW"
    logger.info("predict hour=%d dow=%d wx=%s -> vol=%.0f p=%.3f (%s)",
                inp.hour, inp.day_of_week, inp.weather_main, volume, proba, risk)
    return Prediction(predicted_traffic_volume=round(volume, 1),
                      congestion_probability=round(proba, 4),
                      congestion_predicted=proba >= 0.5,
                      risk_level=risk)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

