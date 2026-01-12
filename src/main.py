from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware

from .schema import PredictedRequest, PredictionResponse
from .data_fetcher import fetch_last_60_days
from .prediction_engine import predict_next_close, calculate_confidence

app = FastAPI(title="Crypto Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow all frontends (OK for dev)
    allow_credentials=True,
    allow_methods=["*"],        # allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],        # allow JSON headers
)

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
