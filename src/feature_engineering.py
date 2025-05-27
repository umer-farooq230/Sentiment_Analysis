from src.data_processing import load_and_clean

path = "E:\\Umer\\projects\\sentiment analysis\\google_review_model\\data\\raw\\googleplaystore_user_reviews.csv"
df = load_and_clean(path)

def app_review_count(df):
    df["app_review_Count"] = df.groupby("App")["App"].transform("count")
    return df

def word_count_and_char_count(df):
    df["word_count"] = df["Translated_Review"].str.split().apply(len)
    df["char_count"] = df["Translated_Review"].str.len()
    return df

def exclamations_and_question_mark(df):    
    df["exclamations"] = df["Translated_Review"].str.count("r\!")
    df["question_mark"] = df["Translated_Review"].str.count("r\?")

    return df
    
def mismatch_flag(df):
    df["mismatch_flag"] = [True if p*s < 0 else False for p,s in zip(df["Sentiment_Polarity"] , df["Sentiment_Subjectivity"])]  
    return df

feature_function = [app_review_count, word_count_and_char_count, exclamations_and_question_mark, mismatch_flag]
 
#print(df.columns)
#print(df.head())