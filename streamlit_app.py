import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

#  UI
st.title("Sentiment Analysis WebApp")
st.write("Please train the model first. Then, enter feedback to analyze its sentiment.")

# Initialize session state for model storage
# Without this, the App can't use text_clf for predictions.
if "text_clf" not in st.session_state:
    st.session_state.text_clf = None

# Load and process training data
def train_model():
    df_train = pd.read_csv("train.csv", encoding='latin-1')
    df_train.dropna(inplace=True)
    X_train = df_train['text']
    y_train = df_train['sentiment']

    # Create and train model
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC())
    ])
    text_clf.fit(X_train, y_train)

    # Store model in session state
    st.session_state.text_clf = text_clf  
    st.success("Model trained successfully!")

# Train Model Button
if st.button("Train Model"):
    train_model()

# User input for sentiment analysis
text_input = st.text_input("Enter your feedback:")

# Disable Analyze Sentiment button if model is not trained
analyze_button = st.button("Analyze Sentiment", disabled=st.session_state.text_clf is None)

# Perform sentiment analysis
if analyze_button and text_input:
    text_clf = st.session_state.text_clf
    prediction = text_clf.predict([text_input])[0]

    # Color-coded output
    color = "green" if prediction == "positive" else "yellow" if prediction == "neutral" else "red"
    st.markdown(f"<p style='color:{color}; font-size:20px;'>Predicted Sentiment: {prediction}</p>", unsafe_allow_html=True)
