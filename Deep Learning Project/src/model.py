import tensorflow as tf
from tensorflow.keras import layers

def build_model(vocab_size, embedding_dim, max_sequence_length, lstm_units=64, dropout_rate=0.5):
    model = tf.keras.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
        layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Bidirectional(layers.LSTM(lstm_units)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.build(input_shape=(None, max_sequence_length))
    model.summary()
    return model
