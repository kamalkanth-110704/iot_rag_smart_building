# main.py

import pandas as pd
from anomaly_detection import detect_anomalies
from predictive_maintenance import predict_failure
from rag_query import rag_query

def run_system():
    # Load dataset
    df = pd.read_csv("cleaned_iot_dataset.csv")

    # Pick a sample reading (simulate real-time)
    for i in range(0, len(df), 10000):  # Every 10,000th reading
        sample = df.iloc[i]
        sensor_data = {
            "temperature": sample["temperature"],
            "humidity": sample["humidity"],
            "use [kW]": sample["use [kW]"],
            "vibration": sample["vibration"]
        }

        print("\n📡 Sensor Reading:", sensor_data)

        
        alerts = detect_anomalies(sensor_data)
        if alerts:
            print("⚠ Anomaly Alerts:", alerts)
        else:
            print("✅ No anomalies detected")

        
        failure_prob = predict_failure(sensor_data)
        print(f" Failure Probability: {failure_prob * 100:.2f}%")

        
        if failure_prob > 0.5:
            question = "How to fix high temperature in HVAC?"
            answer = rag_query(question)
            print(" Maintenance Advice:", answer)

        
        if i > 30000:
            break

if __name__ == "__main__":
    run_system()
