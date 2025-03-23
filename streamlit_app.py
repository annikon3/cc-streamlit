import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# UI
st.title("Sentiment Analysis WebApp")
st.write("Please first train the model. Then, enter a sentence to analyze its sentiment.")

# Load and process training data
def train_model():
    st.session_state.training = True
    st.session_state.feedback = "Please wait, training in progress..."
    # Disable butto nwhile trainign in progrsss
    st.session_state.disable_train_button = True
    st.rerun()
    
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
    
    st.session_state.training = False
    st.session_state.feedback = "Model trained successfully!"
    # Enable button
    st.session_state.disable_train_button = False
    st.rerun()

if "training" not in st.session_state:
    st.session_state.training = False
    st.session_state.feedback = ""
    st.session_state.disable_train_button = False

if st.button("Train Model", disabled=st.session_state.disable_train_button):
    train_model()

st.write(st.session_state.feedback)

# Text input for entering feedback
text_input = st.text_input("Enter your feedback:")
if st.button("Analyze Feedback Sentiment") and text_input:
    analysis = text_clf.predict([text_input])[0]
    st.write("Analyzed Sentiment:", analysis)
