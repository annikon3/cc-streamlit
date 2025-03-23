import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Load and preprocess training data
df_train = pd.read_csv("train.csv", encoding='latin-1')
df_train.dropna(inplace=True)
X_train = df_train['text']
y_train = df_train['sentiment']

# Create and train the model
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])
text_clf.fit(X_train, y_train)

# Streamlit UI
st.title("Sentiment Analysis WebApp")
st.write("Enter a sentence to analyze its sentiment.")

text_input = st.text_input("Enter your text:")
if text_input:
    prediction = text_clf.predict([text_input])[0]
    st.write("Predicted Sentiment:", prediction)

