import numpy as np
import pickle
from keras.models import load_model
from pathlib import Path
import warnings

BASE_DIR = Path(__file__).resolve().parents[2]  # adjust if needed
MODELS_DIR = BASE_DIR / "models"

MODELS = {}
SCALERS = {}

FEATURE_CLOSE = 3  # OHLC → close index

def load_assets(symbol: str):
    """Load model and scaler for a given symbol."""
    if symbol not in MODELS:
        model_path = MODELS_DIR / f"{symbol}_gru.keras"
        scaler_path = MODELS_DIR / f"{symbol}_scaler.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        MODELS[symbol] = load_model(model_path)
        with open(scaler_path, "rb") as f:
            SCALERS[symbol] = pickle.load(f)

    return MODELS[symbol], SCALERS[symbol]


def preprocess(data: np.ndarray, scaler):
    """Scale data and reshape for model input."""
    if data.shape != (60, 4):
        raise ValueError(f"Input must be shape (60,4), got {data.shape}")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
        scaled = scaler.transform(data)
    return scaled.reshape(1, 60, 4)


def inverse_close(value_scaled: float, scaler):
    """Inverse scaling for the close value."""
    dummy = np.zeros((1, scaler.n_features_in_))
    dummy[0, FEATURE_CLOSE] = value_scaled
    return scaler.inverse_transform(dummy)[0, FEATURE_CLOSE]


def predict_next_close(symbol: str, last_60_days: np.ndarray):
    """Predict the next close price for a given crypto symbol."""
    model, scaler = load_assets(symbol)
    X = preprocess(last_60_days, scaler)
    y_scaled = model.predict(X, verbose=0)[0][0]
    return float(inverse_close(y_scaled, scaler))


def calculate_confidence(last_60_days: np.ndarray):
    """Calculate a confidence score based on recent volatility."""
    close = last_60_days[:, FEATURE_CLOSE]
    returns = np.diff(close) / close[:-1]
    volatility = np.std(returns)
    confidence = 100 - volatility * 1200
    return round(max(50, min(confidence, 95)), 2)


