import joblib
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved Random Forest model
model_filename = 'random_forest_model.joblib'
random_forest = joblib.load(model_filename)

# Load the saved TF-IDF vectorizer
tfidf_vectorizer_filename = 'tfidf_vectorizer.joblib'
tfidf_vectorizer = joblib.load(tfidf_vectorizer_filename)

# Input from the user in the form of a long text
user_input = input("Enter text for feedback prediction: ")
nlp = spacy.load("en_core_web_sm")
import string

def remove_stopwords_and_punctuation(text):
    doc = nlp(text)
    filtered_text = [token.text for token in doc if not (token.is_stop or token.text in string.punctuation)]
    return ' '.join(filtered_text)

processed = remove_stopwords_and_punctuation(user_input)
input_tfidf = tfidf_vectorizer.transform(pd.Series(processed))

# Use the loaded model to predict the feedback class
predicted_feedback = random_forest.predict(input_tfidf)


