import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# UI
st.title("Sentiment Analysis WebApp")
st.write("Enter feedback to analyze its sentiment.")

# Load and process training data
def train_model():
    df_train = pd.read_csv("train.csv", encoding='latin-1')
    df_train.dropna(inplace=True)
    X_train = df_train['text']
    y_train = df_train['sentiment']

    global text_clf
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC())
    ])
    text_clf.fit(X_train, y_train)
    st.success("Model trained successfully!")

if st.button("Train Model"):
    train_model()

text_input = st.text_input("Please enter your feedback:")

if st.button("Analyze Sentiment") and text_input:
    prediction = text_clf.predict([text_input])[0]
    
    color = "black"
    if prediction == "negative":
        color = "red"
    elif prediction == "neutral":
        color = "yellow"
    elif prediction == "positive":
        color = "green"
    
    st.markdown(f"<p style='color:{color}; font-size:20px;'>Predicted Sentiment: {prediction}</p>", unsafe_allow_html=True)
    