# Google Play Store Sentiment Analysis (End-to-End Pipeline)

This project is an end-to-end pipeline for **sentiment analysis on user reviews** from the **Google Play Store** dataset. It covers everything from raw data to model evaluation — using clean modular code, feature engineering, model training, and evaluation.

---

## Project Goal

To build a robust pipeline that predicts the **sentiment (Positive, Negative, Neutral)** of user reviews using real-world noisy data. We also explore how feature engineering and proper pipeline structuring can improve modeling on messy, semi-structured text data.

---

## Project Structure

```
google_review_model/
│
├── config.yaml                # Config file 
├── data/
│   ├── raw/                   raw dataset
│
├── src/
|   ├── data_info.ipynb
│   ├── data_processing.py     # Loading and cleaning the dataset
│   ├── feature_engineering.py # new features
│   ├── model.py               # Model training script 
│   ├── model.pkl              # Saved trained model
│   ├── label_encoder.pkl      # Saved label encoder
│
│
└── README.md                  # This file
```


---

## Feature Engineering Highlights

We engineered several meaningful features such as:

* **Review length**
* **Polarity/subjectivity thresholds**
* **App frequency** (how many reviews per app)
* **Capitalization ratio**
* **Exclamation density**

These features were designed to ask:

* Do longer reviews lean more negative?
* Is highly subjective content tied to a sentiment?
* Are users who use more exclamation marks angrier?

---

## Getting Started

### 1. Install dependencies

Make sure you have Python 3.8+ installed. Then:

```bash
pip install -r requirements.txt
```

### 2. Set paths in `config.yaml`

Update your config file with local paths if needed:

```yaml
data:
  raw_path: data/raw/googleplaystore_user_reviews.csv
  
model:
  save_path: models/model.pkl
  encoder_path: models/label_encoder.pkl
```

### 3. Run the training pipeline

```bash
python src/model.py
```

This will load the data, clean it, apply feature engineering, train the model, and print evaluation results.

### 4. Run separate evaluation

```bash
python evaluate.py
```

---

## Sample Output

```
              precision    recall  f1-score   support

    Negative       0.80      0.76      0.78       350
     Neutral       0.60      0.64      0.62       300
    Positive       0.88      0.85      0.86       400

    accuracy                           0.77      1050
```

---

## Technologies Used

* Python 3
* XGBoost
* scikit-learn
* Pandas, NumPy
* Joblib (for saving models)
* YAML (for configs)

---

## What's Next?

* Add a simple UI (e.g., Streamlit app)
* Experiment with NLP techniques (TF-IDF, embeddings)
* Serve as API using FastAPI
* Use MLflow or Weights & Biases for experiment tracking

---

## Author

Made with 💻 by \Umer

