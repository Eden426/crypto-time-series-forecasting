from datetime import datetime, timezone
from threading import Lock
from time import monotonic

import numpy as np
import requests

COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "LTC": "litecoin",
    "XRP": "ripple",
}
BINANCE_SYMBOLS = {symbol: f"{symbol}USDT" for symbol in COINGECKO_IDS}
COINBASE_PRODUCTS = {symbol: f"{symbol}-USD" for symbol in COINGECKO_IDS}
HEADERS = {"User-Agent": "crypto-prediction-api/1.0"}
CACHE_TTL_SECONDS = 300
_CACHE: dict[str, tuple[float, np.ndarray]] = {}
_CACHE_LOCK = Lock()


def _validate_ohlc(rows: list[list[float]]) -> np.ndarray:
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (60, 4) or not np.isfinite(result).all():
        raise RuntimeError("Market data provider did not return 60 valid OHLC rows.")
    return result


def _fetch_coingecko(symbol: str) -> np.ndarray:
    coin_id = COINGECKO_IDS[symbol]
    response = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
        params={"vs_currency": "usd", "days": 90},
        headers=HEADERS,
        timeout=12,
    )
    if response.status_code != 200:
        raise RuntimeError(f"CoinGecko returned HTTP {response.status_code}.")

    daily_prices: dict[str, list[float]] = {}
    for timestamp_ms, price in response.json().get("prices", []):
        day = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).date().isoformat()
        daily_prices.setdefault(day, []).append(float(price))
    rows = [
        [values[0], max(values), min(values), values[-1]]
        for _, values in sorted(daily_prices.items())
    ][-60:]
    return _validate_ohlc(rows)


def _fetch_binance(symbol: str) -> np.ndarray:
    response = requests.get(
        "https://data-api.binance.vision/api/v3/klines",
        params={"symbol": BINANCE_SYMBOLS[symbol], "interval": "1d", "limit": 60},
        headers=HEADERS,
        timeout=12,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Binance market data returned HTTP {response.status_code}.")
    rows = [[float(candle[index]) for index in (1, 2, 3, 4)] for candle in response.json()]
    return _validate_ohlc(rows)


def _fetch_coinbase(symbol: str) -> np.ndarray:
    """Fetch daily USD candles from Coinbase Exchange as a second fallback."""
    now = datetime.now(timezone.utc)
    start = datetime.fromtimestamp(now.timestamp() - 70 * 24 * 60 * 60, tz=timezone.utc)
    response = requests.get(
        f"https://api.exchange.coinbase.com/products/{COINBASE_PRODUCTS[symbol]}/candles",
        params={"granularity": 86400, "start": start.isoformat(), "end": now.isoformat()},
        headers={**HEADERS, "Accept": "application/json"},
        timeout=12,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Coinbase market data returned HTTP {response.status_code}.")

    # Coinbase returns [time, low, high, open, close, volume] newest first.
    candles = sorted(response.json(), key=lambda candle: candle[0])
    rows = [
        [float(candle[3]), float(candle[2]), float(candle[1]), float(candle[4])]
        for candle in candles
        if len(candle) >= 5
    ][-60:]
    return _validate_ohlc(rows)


def fetch_last_60_days(symbol: str) -> np.ndarray:
    """Return 60 daily OHLC rows, using cached data and provider fallbacks."""
    symbol = symbol.upper()
    if symbol not in COINGECKO_IDS:
        raise ValueError(f"Unsupported symbol: {symbol}")

    # Serialize cache misses so a burst of requests cannot consume the data provider quota.
    with _CACHE_LOCK:
        cached = _CACHE.get(symbol)
        if cached and cached[0] > monotonic():
            return cached[1].copy()

        errors = []
        last_error = None
        for provider in (_fetch_coingecko, _fetch_binance, _fetch_coinbase):
            try:
                result = provider(symbol)
                break
            except Exception as error:
                last_error = error
                errors.append(f"{provider.__name__.removeprefix('_fetch_')} failed ({error})")
        else:
            raise RuntimeError("; ".join(errors)) from last_error

        _CACHE[symbol] = (monotonic() + CACHE_TTL_SECONDS, result.copy())
        return result
