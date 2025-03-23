import streamlit as st
from textblob import TextBlob

st.title("Sentiment Analysis WebApp")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
text = st.text_area("Please Enter your text")

if st.button("Analyze "): 
    blob = TextBlob(message) 
    result = blob.sentiment 
    polarity = result.polarity 
    subjectivity = result.subjectivity 
    if polarity < 0: 
        st.warning("The entered text has negative sentiments associated with it"+str(polarity)) 
        rain( 
        emoji="????", 
        font_size=20, # the size of emoji 
        falling_speed=3, # speed of raining 
        animation_length="infinite", # for how much time the animation will happen 
    ) 
    if polarity >= 0: 
        st.success("The entered text has positive sentiments associated with it."+str(polarity)) 
        rain( 
        emoji="????", 
        font_size=20, # the size of emoji 
        falling_speed=3, # speed of raining 
        animation_length="infinite", # for how much time the animation will happen 
        ) 
    st.success(result) 


# Implement a UI with Streamlit for Sentiment Analysis​
""" 
Objective: Utilize the Streamlit framework to create an interactive user interface (UI) for sentiment analysis, and deploy this application on Streamlit Cloud​

Key Actions:​

Develop an application using Streamlit that allows users to input text and submit it for sentiment analysis​
The application should dynamically display the sentiment analysis results using Streamlit's interactive widgets and data visualization capabilities​
Address the task in the report
Deployment Target: Streamlit Cloud, which is specifically optimized for hosting Streamlit applications
 """

