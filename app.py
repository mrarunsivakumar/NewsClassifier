#import libraries
import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder

# Adding a red-colored title
title_html = """
<h1 style="color: red;">NewsClassifier: Building an Automated News
Classification System with NLP Techniques</h1>
"""
st.markdown(title_html, unsafe_allow_html=True)

# Load the saved model 
with open('NewsClassifier.pkl', 'rb') as model_file:
    svm_pipeline = pickle.load(model_file)

# Load the saved label encoder
with open('NewsClassifier_label_encoder.pkl', 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

# User input for news content
content_input = st.text_area('Enter News Content:', '')

if st.button('Classify'):
    # Make prediction
    y_pred_svm_code = svm_pipeline.predict([content_input])[0]

    # Map code to topic name
    predicted_topic = label_encoder.inverse_transform([y_pred_svm_code])[0]

    # Display result
    st.success(f'Predicted Topic: {predicted_topic}')

