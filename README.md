# Crypto Time-Series Forecasting

## Project Overview

This project implements a **machine learning–based cryptocurrency price forecasting system** using historical time-series data. 
The system predicts the **next-day closing price** of selected cryptocurrencies (BTC, ETH, LTC, XRP) using deep learning models such as **LSTM** and **GRU**.

The project consists of three main parts:

* **Model training & evaluation** (offline)
* **Backend API** for inference (FastAPI)
* **Frontend web interface** for user interaction

---

## Final Project Structure

```
Crypto-Time-Series-Forecasting/
│
├── backend/
│   ├── src/
│   │   ├── data_fetcher.py        # Fetches last 60 days market data (CoinGecko)
│   │   ├── prediction_engine.py   # Loads models, predicts price & confidence
│   │   ├── schema.py              # API request/response schemas
│   │   ├── main.py                # FastAPI entry point
│   │   └── __init__.py
│
├── frontend/
│   └── src/
│       ├── index.html             # UI layout
│       ├── style.css              # Styling
│       └── index.js               # API calls & UI logic
│
├── dataset/
│   └── raw/
│       └── crypto_combine.csv     # Historical dataset used for training
│
├── models/                        # Final trained models
│   ├── BTC_gru.keras
│   ├── ETH_gru.keras
│   ├── LTC_gru.keras
│   ├── XRP_gru.keras
│   ├── BTC_scaler.pkl
│   ├── ETH_scaler.pkl
│   ├── LTC_scaler.pkl
│   └── XRP_scaler.pkl
│
├── src/                           # Model training & evaluation (offline)
│   ├── data/
│   │   ├── load_dataset.py
│   │   ├── preprocess.py
│   │   └── main.py
│   ├── data_analysis/
│   │   └── analysis_dst.py
│   ├── models/
│   │   ├── train.py
│   │   ├── model_train.py
│   │   └── evaluate.py
│   └── predict/
│
├── requirements.txt               # ML & training dependencies
└── README.md
```

---

## Dataset Description

* **Source:** Historical cryptocurrency price data (CSV)
* **Features:** Open, High, Low, Close (OHLC)
* **Frequency:** Daily
* **Input Window:** Last **60 days**
* **Target:** **Next-day Close price**

---

## Feature Engineering

Feature engineering prepares raw time-series data into a format suitable for deep learning models.

Steps used in this project:

* Selection of OHLC features
* Normalization using **MinMaxScaler**
* Creation of **sliding windows (60 timesteps)**
* Reshaping into `(samples, timesteps, features)`

No manual user input is required — the model learns patterns directly from historical trends.

---

## Model Details

* **Models:** GRU (primary), LSTM (experimental)
* **Framework:** TensorFlow / Keras
* **Input Shape:** `(60, 4)` → 60 days × OHLC
* **Output:** Single scalar → next-day Close price
* **Loss Function:** Mean Squared Error (MSE)
* **Optimizer:** Adam

### Why GRU?

GRU was selected for deployment due to:

* Faster training than LSTM
* Lower memory usage
* Comparable or better performance on this dataset

---

## Model Evaluation Results (Summary)

| Asset | RMSE  | MAE   |
| ----- | ----- | ----- |
| BTC   | ~1178 | ~790  |
| ETH   | ~85   | ~60   |
| LTC   | ~4.1  | ~3.0  |
| XRP   | ~0.02 | ~0.01 |

Lower error values indicate better short-term trend learning, especially for lower-volatility assets.

---

## 🔌 API Endpoints

### Base URL

```
http://127.0.0.1:8000
```

### POST /predict

Predicts the next-day closing price.

**Request Body**

```json
{
  "symbol": "BTC"
}
```

**Response**

```json
{
  "symbol": "BTC",
  "predicted_close": 31522.78,
  "confidence": 72,
 "description": "Next-day close prediction for BTC"}
```

---

## How It Works (End-to-End)

1. User selects a cryptocurrency in the frontend
2. Frontend sends request to FastAPI backend
3. Backend fetches last 60 days of market data
4. Data is normalized using the saved scaler
5. GRU model predicts next-day Close price
6. Confidence score is calculated using recent volatility
7. Prediction is returned and displayed in UI

---
📽️ Demo Video: https://drive.google.com/file/d/1gmuNEZky17c4u_2l8W0_hA8sK6M_WuWb/view?usp=sharing


## Requirements

### Machine Learning & Data Processing
The following libraries are used for data preprocessing, model training, and evaluation:

- Python 3.9+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- TensorFlow / Keras

### Backend (API & Inference)
The backend service is built using:

- FastAPI
- Uvicorn
- Pydantic
- Python-multipart
- Requests

### Frontend
- HTML
- CSS
- JavaScript (Vanilla)

>  Note: Exact dependency versions are specified in:
> - `requirements.txt` (model training & evaluation)
> - `backend/requirements.txt` (API & inference)

All dependencies can be installed using pip and the provided requirements files.


## Running the Project
### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
uvicorn src.main:app --reload
```

### Frontend

Open directly in browser:

```
frontend/src/index.html
```

Ensure backend is running before prediction.

---

## Notes & Limitations

* Predicts **only the next-day Close price**
* No real-time streaming data (uses recent historical data)
* Confidence is **heuristic**, not probabilistic
* Model performance may degrade during extreme market events

---

## 🔮 Future Enhancements

* Multi-output prediction (Open, High, Low, Close)
* Technical indicators (RSI, MACD)
* Interactive charts and trend visualization
* Historical accuracy tracking
* Model comparison dashboard
* Improved confidence estimation

---

## Disclaimer

This project is for **educational purposes only** and does **not** constitute financial advice.

---

## Authors
Eden Nigatu