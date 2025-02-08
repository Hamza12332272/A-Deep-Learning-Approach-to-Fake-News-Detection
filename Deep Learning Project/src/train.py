from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
import numpy as np

def train_model(model, dataset, max_sequence_length, epochs=5, batch_size=64):
    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]

    # Calculate class weights
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    early_stopping = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], class_weight=class_weights)
    return model, X_test, y_test
