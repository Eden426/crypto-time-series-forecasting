---
library_name: keras
tags:
  - crypto
  - time-series
  - forecasting
  - gru
  - tensorflow
license: mit
---

# Crypto Time-Series Forecasting GRU Models

This repository contains trained **Gated Recurrent Unit (GRU)** deep learning models and `MinMaxScaler` artifacts for predicting the **next-day closing price (USD)** of four cryptocurrencies:
- **BTC** (Bitcoin)
- **ETH** (Ethereum)
- **LTC** (Litecoin)
- **XRP** (Ripple)

## Files Included

| Asset | Model File | Scaler File | RMSE | MAE |
| :--- | :--- | :--- | :--- | :--- |
| **BTC** | `BTC_gru.keras` | `BTC_scaler.pkl` | ~1178 | ~790 |
| **ETH** | `ETH_gru.keras` | `ETH_scaler.pkl` | ~85 | ~60 |
| **LTC** | `LTC_gru.keras` | `LTC_scaler.pkl` | ~4.1 | ~3.0 |
| **XRP** | `XRP_gru.keras` | `XRP_scaler.pkl` | ~0.02 | ~0.01 |

## Model Architecture

- **Input Shape:** `(60, 4)` — 60 days of `[Open, High, Low, Close]`
- **Layers:**
  - `GRU(64, return_sequences=True)`
  - `Dropout(0.2)`
  - `GRU(32)`
  - `Dropout(0.2)`
  - `Dense(1)`
- **Optimizer:** `Adam(learning_rate=0.0005)`
- **Loss Function:** Mean Squared Error (`mse`)

## Usage with `huggingface_hub`

```python
import pickle
import numpy as np
from keras.models import load_model
from huggingface_hub import hf_hub_download

REPO_ID = "Eden-N/crypto-time-series-gru"
symbol = "BTC"

model_path = hf_hub_download(repo_id=REPO_ID, filename=f"{symbol}_gru.keras")
scaler_path = hf_hub_download(repo_id=REPO_ID, filename=f"{symbol}_scaler.pkl")

model = load_model(model_path)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
```
