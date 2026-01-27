import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle

from keras import backend as K

from src.data.load_dataset import load_dataset
from src.data.preprocess import preprocess_cleands
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.predict.predict_ds import predict


def inverse_close(y_scaled, scaler):
    """Convert scaled Close values back to real prices"""
    dummy = np.zeros((len(y_scaled), scaler.n_features_in_))
    dummy[:, -1] = y_scaled.flatten()
    return scaler.inverse_transform(dummy)[:, -1]


def main():
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_PATH = BASE_DIR / "dataset" / "raw" / "crypto_combine.csv"

    raw_data = pd.read_csv(DATA_PATH)
    clean_data = load_dataset(raw_data)

    results = []

    model_dir = BASE_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for crypto, df_crypto in clean_data.groupby("Crypto"):
        print(f" Training model for {crypto}")

        symbol = crypto

        X, y, scaler = preprocess_cleands(df_crypto, window_size=60)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model, history = train_model(
            X_train, y_train,
            X_test, y_test,
            window_size=X.shape[1],
            n_features=X.shape[2],
            epochs=20,
            batch_size=32,
        )

        model_path = model_dir / f"{symbol}_gru.keras"
        model.save(model_path)

        scaler_path = model_dir / f"{symbol}_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        print(f"Saved model to: {model_path}")
        print(f"Saved scaler to: {scaler_path}")

        mse, rmse, mae = evaluate_model(model, X_test, y_test, scaler)
        results.append((crypto, mse, rmse, mae))

        preds_real = predict(X_test, scaler, model)
        y_real = inverse_close(y_test, scaler)

        print(f"{crypto} predictions (first 5):")
        print(preds_real[:5])

        plt.figure(figsize=(10, 4))
        plt.plot(y_real[:100], label="Actual", linewidth=2)
        plt.plot(preds_real[:100], label="Predicted", linestyle="--")
        plt.title(f"{crypto} Price Prediction")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

        K.clear_session()

    for crypto, mse, rmse, mae in results:
        print(f"{crypto} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")


if __name__ == "__main__":
    main()
