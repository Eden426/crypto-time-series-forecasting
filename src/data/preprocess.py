# deep learning models expect NumPy arrays
import numpy as np
# neural networks work best when inputs are scaled
from sklearn.preprocessing import MinMaxScaler
"""
This preprocessing function slides a 60-day window over OHLC data, scales it, 
and converts it into sequences so a GRU can learn temporal patterns and 
predict the next day’s Close price
"""
def preprocess_cleands(data, window_size=60):
    features = ["Open","High","Low","Close"]
    scaler = MinMaxScaler()
    scaled_data= scaler.fit_transform(data[features])
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
         X.append(scaled_data[i-window_size:i])
         y.append(scaled_data[i, features.index("Close")])

    return np.array(X), np.array(y), scaler




