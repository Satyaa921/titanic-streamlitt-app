
import streamlit as st
import pickle
import numpy as np

st.title("Titanic Survival Prediction App")

user_input = st.text_input("Enter feature values (comma separated):")

if st.button("Predict"):
    try:
        input_data = [float(x) for x in user_input.split(",")]
        with open("logistic_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)
        result = "Survived" if prediction[0] == 1 else "Not Survived"
        st.success(f"Prediction: {result}")
    except:
        st.error("Please enter valid numeric values separated by commas.")
