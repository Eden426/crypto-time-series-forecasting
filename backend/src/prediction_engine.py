import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pickle
from keras.models import load_model
from pathlib import Path
import warnings

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

HF_REPO_ID = os.getenv("HF_REPO_ID", "Eden-N/crypto-time-series-gru").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None

MODELS = {}
SCALERS = {}

FEATURE_CLOSE = 3  # OHLC → close index


def _resolve_asset_paths(symbol: str) -> tuple[Path, Path]:
    """Resolve model and scaler paths from Hugging Face Hub (if HF_REPO_ID is set) or local disk."""
    model_filename = f"{symbol}_gru.keras"
    scaler_filename = f"{symbol}_scaler.pkl"

    if HF_REPO_ID:
        from huggingface_hub import hf_hub_download

        model_path = Path(
            hf_hub_download(repo_id=HF_REPO_ID, filename=model_filename, token=HF_TOKEN)
        )
        scaler_path = Path(
            hf_hub_download(repo_id=HF_REPO_ID, filename=scaler_filename, token=HF_TOKEN)
        )
        return model_path, scaler_path

    model_path = MODELS_DIR / model_filename
    scaler_path = MODELS_DIR / scaler_filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Set HF_REPO_ID env var to load from Hugging Face Hub."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. Set HF_REPO_ID env var to load from Hugging Face Hub."
        )

    return model_path, scaler_path


def load_assets(symbol: str):
    """Load model and scaler for a given symbol."""
    if symbol not in MODELS:
        model_path, scaler_path = _resolve_asset_paths(symbol)
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


