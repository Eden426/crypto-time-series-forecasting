import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

from .data_fetcher import fetch_last_60_days
from .prediction_engine import (
    HF_REPO_ID,
    calculate_confidence,
    predict_next_close,
)
from .schema import PredictedRequest, PredictionResponse

logger = logging.getLogger(__name__)
app = FastAPI(title="Crypto Prediction API")
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend" / "src"

raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allowed_origins != ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Crypto Prediction API",
        "model_source": f"hf://{HF_REPO_ID}" if HF_REPO_ID else "local",
        "revision": os.getenv("RENDER_GIT_COMMIT", "local"),
    }


@app.get("/api/config", include_in_schema=False)
def frontend_config():
    return {"sameOrigin": True}


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(FRONTEND_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/style.css", include_in_schema=False)
def frontend_styles():
    return FileResponse(
        FRONTEND_DIR / "style.css", media_type="text/css", headers={"Cache-Control": "no-cache"}
    )


@app.get("/index.js", include_in_schema=False)
def frontend_script():
    return FileResponse(
        FRONTEND_DIR / "index.js", media_type="text/javascript", headers={"Cache-Control": "no-cache"}
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_crypto(request: PredictedRequest):
    symbol = request.symbol.upper()
    try:
        daily_ohlc = fetch_last_60_days(symbol)
        predicted = predict_next_close(symbol, daily_ohlc)
        confidence = calculate_confidence(daily_ohlc)
        return PredictionResponse(
            symbol=symbol,
            predicted_close=round(predicted, 2),
            confidence=confidence,
            history=[round(float(price), 8) for price in daily_ohlc[-30:, 3]],
            description=f"Next-day close prediction for {symbol}",
        )
    except Exception as exc:
        logger.exception("Prediction failed for %s", symbol)
        raise HTTPException(
            status_code=503,
            detail=f"Prediction failed for {symbol}: {exc}",
        ) from exc
