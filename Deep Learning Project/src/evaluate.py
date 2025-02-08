from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test):
    predictions = (model.predict(X_test) > 0.5).astype("int32")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Fake', 'Real']))
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
def analyze_confidence(predictions, threshold=0.9):
    """
    Analyze model confidence for predictions above a certain threshold.
    """
    confident_preds = [pred for pred in predictions if pred > threshold or pred < (1 - threshold)]
    print(f"Confident predictions: {len(confident_preds)} out of {len(predictions)}")
    return confident_preds
