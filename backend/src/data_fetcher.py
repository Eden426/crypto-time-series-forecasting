from datetime import datetime, timezone

import numpy as np
import requests

COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "LTC": "litecoin",
    "XRP": "ripple",
}

HEADERS = {"User-Agent": "crypto-prediction-api/1.0"}


def fetch_last_60_days(symbol: str) -> np.ndarray:
    """Fetch and aggregate market prices into the model's 60 daily OHLC rows."""
    coin_id = COINGECKO_IDS.get(symbol.upper())
    if not coin_id:
        raise ValueError(f"Unsupported symbol: {symbol}")

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    response = requests.get(
        url,
        params={"vs_currency": "usd", "days": 90},
        headers=HEADERS,
        timeout=20,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Market data provider returned HTTP {response.status_code}. Please retry later."
        )

    prices = response.json().get("prices", [])
    daily_prices: dict[str, list[float]] = {}
    for timestamp_ms, price in prices:
        day = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).date().isoformat()
        daily_prices.setdefault(day, []).append(float(price))

    rows = [
        [values[0], max(values), min(values), values[-1]]
        for _, values in sorted(daily_prices.items())
    ][-60:]
    if len(rows) < 60:
        raise RuntimeError(
            f"Market data provider returned only {len(rows)} daily points; 60 are required."
        )
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (60, 4) or not np.isfinite(result).all():
        raise RuntimeError("Market data provider returned invalid price data.")
    return result
