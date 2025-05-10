import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Titanic Survival Predictor")

# UI elements for each input
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
age = st.slider("Age", 0, 80, 30)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.slider("Fare", 0.0, 500.0, 32.2)
sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Manual encoding for categorical values (must match model training)
sex_val = 1 if sex == "male" else 0
embarked_val = {"S": 2, "C": 0, "Q": 1}[embarked]

# Collect input features
input_data = np.array([[pclass, age, sibsp, parch, fare, sex_val, embarked_val]])

# Predict
if st.button("Predict Survival"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    result = "Survived" if prediction == 1 else "Not Survived"
    st.success(f"Prediction: {result}")
