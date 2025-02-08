import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
from nltk.corpus import wordnet
import pickle


def load_data(fake_path, true_path):
    """
    Load and merge the Fake and True datasets, adding labels.
    """
    fake_data = pd.read_csv(fake_path)
    true_data = pd.read_csv(true_path)

    # Add labels: 0 for Fake, 1 for True
    fake_data['label'] = 0
    true_data['label'] = 1

    # Concatenate and shuffle the data
    data = pd.concat([fake_data, true_data]).sample(frac=1).reset_index(drop=True)
    return data


def preprocess_data(data, max_words=10000, max_sequence_length=500):
    """
    Preprocess the dataset for training and testing.
    """
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(data['text'])

    sequences = tokenizer.texts_to_sequences(data['text'])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

    X = padded_sequences
    y = data['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save tokenizer for inference
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "tokenizer": tokenizer
    }


def augment_data(data, num_augments=2):
    """
    Perform data augmentation by replacing words with their synonyms.
    """
    def synonym_replacement(text, n=2):
        words = text.split()
        if len(words) == 0:
            return text
        for _ in range(n):
            idx = random.randint(0, len(words) - 1)
            synonyms = wordnet.synsets(words[idx])
            if synonyms:
                words[idx] = synonyms[0].lemmas()[0].name()
        return ' '.join(words)

    if 'text' not in data or data['text'].isnull().any():
        raise ValueError("The dataset does not contain a valid 'text' column.")

    augmented_texts = [synonym_replacement(text, n=num_augments) for text in data['text'] if isinstance(text, str) and len(text.strip()) > 0]
    return augmented_texts


def adversarial_test(data, num_typos=2):
    """
    Create adversarial test data by introducing typos.
    """
    def add_typos(text, num_typos=2):
        text = list(text)
        for _ in range(num_typos):
            idx = random.randint(0, len(text) - 1)
            text[idx] = random.choice("abcdefghijklmnopqrstuvwxyz")
        return ''.join(text)

    adversarial_texts = [add_typos(text, num_typos=num_typos) for text in data['text']]
    return adversarial_texts
