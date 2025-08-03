# predictive_maintenance.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle

def train_predictive_model():
    """Train a predictive maintenance model with SMOTE oversampling and save it."""
    df = pd.read_csv("cleaned_iot_dataset.csv", low_memory=False)

    X = df[['temperature', 'humidity', 'use [kW]', 'vibration']]
    y = df['failure']

    # Handle missing values (fill NaNs with median)
    X = X.fillna(X.median())

    # Apply SMOTE to balance classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"Original failure ratio: {y.mean():.4f}")
    print(f"After SMOTE failure ratio: {y_resampled.mean():.4f}")
    print(f"Resampled dataset size: {len(X_resampled)}")

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_resampled, y_resampled)

    # Save model
    with open("predictive_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved as predictive_model.pkl")

def predict_failure(sensor_data):
    """Predict failure probability for given sensor reading."""
    with open("predictive_model.pkl", "rb") as f:
        model = pickle.load(f)

    df = pd.DataFrame([sensor_data])
    prob = model.predict_proba(df)[0][1]
    return prob

if __name__ == "__main__":
    # Retrain with SMOTE and missing value handling
    train_predictive_model()

    print("\nEnter sensor readings to predict failure probability:")
    temperature = float(input("Temperature: "))
    humidity = float(input("Humidity: "))
    use_kw = float(input("Use [kW]: "))
    vibration = float(input("Vibration: "))

    normal_prob = predict_failure({
        "temperature": temperature,
        "humidity": humidity,
        "use [kW]": use_kw,
        "vibration": vibration
    })

    extreme_prob = predict_failure({
        "temperature": 150,  # Very high temperature
        "humidity": 10,      # Very low humidity
        "use [kW]": 20,      # High usage
        "vibration": 0.5     # Strong vibration
    })

    print(f"\nFailure probability (your input): {normal_prob:.2f}")
    print(f"Failure probability (extreme stress): {extreme_prob:.2f}")
