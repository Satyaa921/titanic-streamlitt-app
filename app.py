import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Titanic Survival Prediction")
st.write("Enter the 9 features below (comma-separated):")

# Input from user
user_input = st.text_input("Enter values (e.g., 3,22,1,0,7.25,0,1,5,2)")

if st.button("Predict"):
    try:
        input_data = [float(x) for x in user_input.strip().split(',')]
        if len(input_data) != 9:
            st.error("Enter exactly 9 numeric values.")
        else:
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)
            result = "Survived" if prediction[0] == 1 else "Not Survived"
            st.success(f"Prediction: {result}")
    except Exception as e:
        st.error("Invalid input. Please enter 9 numeric values separated by commas.")
