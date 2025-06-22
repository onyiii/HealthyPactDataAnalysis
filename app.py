import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

st.title("My Model Predictor")

# Collect input data from user
feature1 = st.number_input("Enter value for Feature 1")
feature2 = st.number_input("Enter value for Feature 2")

if st.button("Predict"):
    # Format input for model
    input_data = np.array([[feature1, feature2]])  # Adjust shape to match your model
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")
