import joblib
import pandas as pd
from sklearn.metrics import classification_report

def evaluate():
    # Load saved artifacts
    model = joblib.load("src/model.pkl")
    label_encoder = joblib.load("src/label_encoder.pkl")
    x_test, y_test = joblib.load("src/test_set.pkl")

    # Make predictions
    y_preds = model.predict(x_test)
    y_preds_labels = label_encoder.inverse_transform(y_preds)
    y_test_labels = label_encoder.inverse_transform(y_test)

    # Print classification report
    print("Evaluation Results:")
    print(classification_report(y_test_labels, y_preds_labels, target_names=label_encoder.classes_))

if __name__ == "__main__":
    evaluate()
