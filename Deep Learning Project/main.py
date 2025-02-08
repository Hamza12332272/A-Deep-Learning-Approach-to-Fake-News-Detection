import os
from src.preprocess import load_data, preprocess_data, augment_data, adversarial_test
from src.train import train_model
from src.evaluate import evaluate_model, analyze_confidence
from src.model import build_model
from src.optimize import optimize_hyperparameters
from src.baseline import baseline_model
import pickle  # For saving the tokenizer

def get_dataset():
    try:
        import kagglehub
        print("Fetching dataset from Kaggle...")
        dataset_path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
        fake_path = os.path.join(dataset_path, "Fake.csv")
        true_path = os.path.join(dataset_path, "True.csv")
    except ImportError:
        print("kagglehub not installed. Ensure dataset paths are correct.")
        fake_path = r'C:\\Users\\Startklar\\PycharmProjects\\Deep Learning Project\\Data\\fake.csv'
        true_path = r'C:\\Users\\Startklar\\PycharmProjects\\Deep Learning Project\\Data\\true.csv'

    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError("Dataset files not found. Please verify the paths or install kagglehub.")

    return fake_path, true_path


if __name__ == "__main__":
    # Fetch dataset paths
    fake_path, true_path = get_dataset()

    print(f"Using dataset files:\nFake: {fake_path}\nTrue: {true_path}")
    # Load raw data
    print("Loading raw data...")
    raw_data = load_data(fake_path, true_path)

    # Apply data augmentation
    print("Applying data augmentation...")
    augmented_texts = augment_data(raw_data)

    # Generate adversarial test data
    print("Generating adversarial test data...")
    adversarial_texts = adversarial_test(raw_data)

    # Preprocess the data
    print("Preprocessing data...")
    dataset = preprocess_data(raw_data)

    # Save the tokenizer
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(dataset['tokenizer'], f)

    # Run baseline model
    print("Running baseline model...")
    baseline_model(raw_data)

    # Optimize hyperparameters
    print("Optimizing hyperparameters...")
    best_params = optimize_hyperparameters(dataset)
    print("Best hyperparameters:", best_params)

    # Build the model with optimized hyperparameters
    print("Building the model...")
    model = build_model(vocab_size=10000,
                        embedding_dim=best_params['embedding_dim'],
                        max_sequence_length=500,
                        lstm_units=best_params['lstm_units'],
                        dropout_rate=best_params['dropout_rate'])

    # Train the model
    print("Training the model...")
    model, X_test, y_test = train_model(model, dataset, max_sequence_length=500, epochs=5)

    # Save the model for the demo app
    print("Saving the model for demo...")
    model.save("model.h5")

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test)

    # Analyze confidence
    print("Analyzing model confidence...")
    analyze_confidence(model.predict(X_test), threshold=0.9)
