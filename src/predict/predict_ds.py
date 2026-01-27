import numpy as np

def predict(X, scaler, model):
    """
      Predict scaled Close prices,Prepare array for inverse scaling,
      Put predictions in Close column, Convert back to real prices

    """
    preds_scaled = model.predict(X)

    n_features = scaler.n_features_in_
    dummy = np.zeros((len(preds_scaled), n_features))
    dummy[:, -1] = preds_scaled[:, 0]

    real_prices = scaler.inverse_transform(dummy)[:, -1]
    return real_prices
