import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Titanic Survival Prediction App")

# Collect user input
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
age = st.number_input("Age", min_value=0.0, max_value=100.0, step=1.0)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0)
sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
is_alone = st.selectbox("Is Alone", ["Yes", "No"])
title = st.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Other"])

# Encode categorical variables
sex = 0 if sex == "male" else 1
embarked_mapping = {"C": 0, "Q": 1, "S": 2}
embarked = embarked_mapping[embarked]
is_alone = 1 if is_alone == "Yes" else 0
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Other": 4}
title = title_mapping[title]

# Prepare the input array
input_data = np.array([[pclass, age, sibsp, parch, fare, sex, embarked, is_alone, title]])

# Scale the input
input_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_scaled)
result = "Survived" if prediction[0] == 1 else "Not Survived"
st.success(f"Prediction: {result}")
