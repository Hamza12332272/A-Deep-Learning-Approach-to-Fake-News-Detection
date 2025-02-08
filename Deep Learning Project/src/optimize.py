import optuna
from .model import build_model
from .train import train_model

def optimize_hyperparameters(dataset):
    def objective(trial):
        embedding_dim = trial.suggest_int('embedding_dim', 64, 256, step=32)
        lstm_units = trial.suggest_int('lstm_units', 32, 128, step=16)
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.6, step=0.1)

        model = build_model(vocab_size=10000, embedding_dim=embedding_dim, max_sequence_length=500, lstm_units=lstm_units, dropout_rate=dropout_rate)
        model, _, _ = train_model(model, dataset, max_sequence_length=500, epochs=3)
        val_accuracy = model.evaluate(dataset['X_test'], dataset['y_test'], verbose=0)[1]
        return val_accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    return study.best_params
