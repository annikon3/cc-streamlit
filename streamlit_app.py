import streamlit as st
from textblob import TextBlob
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


df_train = pd.read_csv("train.csv", encoding='latin-1')
df_train.dropna(inplace=True)

# Extract features (text) and labels (sentiment)
X_train = df_train['text']
y_train = df_train['sentiment']

# Creating and training the model
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

text_clf.fit(X_train, y_train)

# test data
df_test = pd.read_csv("test.csv", encoding='latin-1')
df_test.head()
X_test = df_test['text']
y_test = df_test['sentiment']

st.title("Sentiment Analysis WebApp")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

text_input = st.text_input(
    "Enter feedback below",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
    placeholder=st.session_state.placeholder,
)

if text_input:
    st.write("You entered: ", text_input)

if st.button("Analyze the Sentiment"): 
  blob = TextBlob(text_input) 
  result = blob.sentiment 
  st.write(result)

text_clf.predict([{text_input}])

# text = st.text_area("Please enter feedback below")
