from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.optimizers import Adam

"""This GRU model uses past 60 days of OHLC data to learn temporal dependencies and
 predict the next closing price using stacked GRU layers and mean squared error loss.
 """
def build_model(window_size, n_features):
    model = Sequential([
    GRU(64, input_shape=(window_size,n_features),
        return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1)

    ])
    model.compile( optimizer=Adam(learning_rate=0.0005),loss="mse")
    return model




