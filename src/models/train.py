# Stops training automatically when the model stops improving.
# Used here to clear memory between training runs.
# Imports the model architecture (GRU layers).respectively
from keras.callbacks import EarlyStopping
from keras import backend as K
from src.models.model_train import build_model

"""
    This function trains a GRU model, stops early if it stops improving, and 
    returns the best version of the model.
"""

def train_model(
    X_train, y_train,
    X_val, y_val,
    window_size,
    n_features,
    epochs=30,
    batch_size=32,
):
    K.clear_session()  # important when looping over cryptos

    model = build_model(window_size, n_features)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    """Train the model on past data, test it on unseen data each epoch, 
    stop early if it stops improving.
    
    Validation data = is used to check learning quality
    while training, without learning from it.
    """

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    return model, history





