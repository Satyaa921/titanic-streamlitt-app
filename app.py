import streamlit as st
import numpy as np
import pickle

# Load the model and scaler
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Titanic Survival Prediction App")

st.write("Please enter the following passenger details:")

# These features must match exactly the order used during training
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
age = st.number_input("Age", min_value=0.0, max_value=100.0, step=1.0)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0)
sex = st.selectbox("Sex (0 = male, 1 = female)", [0, 1])
embarked = st.selectbox("Embarked (0 = C, 1 = Q, 2 = S)", [0, 1, 2])
alone = st.selectbox("IsAlone (0 = No, 1 = Yes)", [0, 1])
title = st.selectbox("Title (0 = Mr, 1 = Miss, 2 = Mrs, etc)", [0, 1, 2, 3, 4])  # Optional: match preprocessing

input_data = [pclass, age, sibsp, parch, fare, sex, embarked, alone, title]

if st.button("Predict"):
    try:
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)

        result = "Survived" if prediction[0] == 1 else "Not Survived"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
