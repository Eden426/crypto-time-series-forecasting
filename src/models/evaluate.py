from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from src.predict.predict_ds import predict

def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate model using real (inverse-scaled) Close prices
    """

    # Use your predict() function
    y_pred = predict(X_test, scaler, model)

    # Convert y_test to real prices
    dummy = np.zeros((len(y_test), scaler.n_features_in_))
    dummy[:, -1] = y_test.flatten()
    y_true = scaler.inverse_transform(dummy)[:, -1]

    # Metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    print("Evaluation Results:")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")

    return mse, rmse, mae
