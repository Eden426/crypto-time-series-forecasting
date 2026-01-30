from pydantic import BaseModel, Field
from typing import Literal

cryptoSymbol = Literal["BTC","ETH","LTC","XRP"]

class PredictedRequest(BaseModel):
    symbol : cryptoSymbol


class PredictionResponse(BaseModel):
    symbol: cryptoSymbol
    predicted_close: float = Field(..., description="Predicted closing price")
    confidence:float = Field(...,ge=0, le=100)
    description: str | None = None


