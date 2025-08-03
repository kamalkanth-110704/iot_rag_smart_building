# app.py
import streamlit as st
from rag_system import query_system

st.set_page_config(page_title="IoT RAG Predictive Maintenance", layout="centered")

st.title("🏢 IoT RAG Predictive Maintenance System")
st.write("Ask maintenance questions and predict equipment failure probability using IoT sensor data.")

# User input: Maintenance question
question = st.text_input("Enter your maintenance question:")

# User input: Sensor data
st.subheader("Sensor Readings")
temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
use_kw = st.number_input("Power Usage (kW)", min_value=0.0, step=0.1)
vibration = st.number_input("Vibration (g)", min_value=0.0, step=0.001)

# Predict button
if st.button("Get Maintenance Advice"):
    if question.strip() == "":
        st.warning("Please enter a maintenance question.")
    else:
        sensor_data = {
            "temperature": temperature,
            "humidity": humidity,
            "use [kW]": use_kw,
            "vibration": vibration
        }
        response = query_system(question, sensor_data)
        st.success("Result:")
        st.write(response)
