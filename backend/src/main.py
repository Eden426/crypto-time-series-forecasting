import os
from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware

from .schema import PredictedRequest, PredictionResponse
from .data_fetcher import fetch_last_60_days
from .prediction_engine import predict_next_close, calculate_confidence, HF_REPO_ID

app = FastAPI(title="Crypto Prediction API")

raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Crypto Prediction API",
        "model_source": f"hf://{HF_REPO_ID}" if HF_REPO_ID else "local",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_crypto(request: PredictedRequest):
    symbol = request.symbol.upper()
    try:
        last_60_days = fetch_last_60_days(symbol)  # make sure fetcher returns dates
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    predicted = predict_next_close(symbol, last_60_days)
    confidence = calculate_confidence(last_60_days)

    return PredictionResponse(
        symbol=symbol,
        predicted_close=round(predicted, 2),
        confidence=confidence,
        description=f"Next-day close prediction for {symbol}",
    )
