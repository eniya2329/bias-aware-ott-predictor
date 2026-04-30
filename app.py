import streamlit as st
import pandas as pd
from src.preprocess import load_data, bias_correction, feature_engineering
from src.model import train_model

st.title("Bias-Aware OTT Content Predictor")

# Load data
df = load_data("data/data.csv")

# Process data
df = bias_correction(df)
df = feature_engineering(df)

# Train model
model, score = train_model(df)

st.write("### Model Accuracy (R² Score):", round(score, 2))

st.write("### Data Preview")
st.dataframe(df)

# User Input
st.write("### Predict New Content")

watch_time = st.number_input("Watch Time", value=10000)
completion_rate = st.slider("Completion Rate", 0.0, 1.0, 0.7)
youtube_views = st.number_input("YouTube Views", value=300000)

if st.button("Predict"):
    prediction = model.predict([[watch_time, completion_rate, youtube_views]])
    st.success(f"Predicted Engagement Score: {round(prediction[0], 2)}")