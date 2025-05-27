import pandas as pd
import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


from src.data_processing import load_and_clean
from src.feature_engineering import feature_function

def train_model(path):

    df = load_and_clean(path=path)

    for function in feature_function:
        df = function(df)

    #print(df.head())
    #print(df.columns)
    x = df.drop(columns=["App", "Translated_Review", "Sentiment"])
    y = df["Sentiment"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2)


    model = xgb.XGBClassifier(num_class=3, objective = "multi:softprob", eval_metric = "mlogloss")
    model.fit(x_train, y_train)

    y_preds = model.predict(x_test)
    y_preds_labels = label_encoder.inverse_transform(y_preds)
    y_test_labels = label_encoder.inverse_transform(y_test)

    
    joblib.dump(model, "src/model.pkl")
    joblib.dump(label_encoder, "src/label_encoder.pkl")
    joblib.dump((x_test, y_test), "src/test_set.pkl")
    return model, label_encoder

if __name__ == "__main__":
    path = "E:\\Umer\\projects\\sentiment analysis\\google_review_model\\data\\raw\\googleplaystore_user_reviews.csv"
    train_model(path=path)