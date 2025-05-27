import pandas as pd 
import numpy as np


def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.dropna()
    return df

path= "E:\\Umer\\projects\\sentiment analysis\\google_review_model\\data\\raw\\googleplaystore_user_reviews.csv"
df = load_and_clean(path)
#print(df.head())
