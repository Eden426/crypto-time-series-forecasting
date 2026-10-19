from pydantic import BaseModel, Field
from typing import Literal

CryptoSymbol = Literal["BTC", "ETH", "LTC", "XRP"]


class PredictedRequest(BaseModel):
    symbol: CryptoSymbol


class PredictionResponse(BaseModel):
    symbol: CryptoSymbol
    predicted_close: float = Field(..., description="Predicted closing price")
    confidence: float = Field(..., ge=0, le=100)
    description: str | None = None
    history: list[float] = Field(default_factory=list, description="Recent daily closes in USD")
