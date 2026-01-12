import requests
import numpy as np

COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "LTC": "litecoin",
    "XRP": "ripple"
}

HEADERS = {
    "User-Agent": "crypto-prediction-api/1.0"
}

def fetch_last_60_days(symbol: str):
    """
    Fetch the last 60 days of OHLC data for a crypto symbol from CoinGecko.
    Returns:
        ohlc: np.ndarray of shape (60, 4) [Open, High, Low, Close] (all same as CoinGecko provides only price)
        dates: list of 60 date strings (YYYY-MM-DD)
    """
    coin_id = COINGECKO_IDS.get(symbol.upper())
    if not coin_id:
        raise ValueError(f"Unsupported symbol: {symbol}")

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": 60
    }

    r = requests.get(url, params=params, headers=HEADERS, timeout=10)
    if r.status_code != 200:
        raise RuntimeError("Market data temporarily unavailable")

    data = r.json()
    prices = data.get("prices", [])[-60:]

    if len(prices) < 60:
        raise RuntimeError("Not enough data points")

    # OHLC array (CoinGecko provides only prices, so we copy to OHLC)
    ohlc = np.array([[p[1], p[1], p[1], p[1]] for p in prices])



    return ohlc
