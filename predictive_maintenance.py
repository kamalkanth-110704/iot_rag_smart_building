# predictive_maintenance.py

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

# File paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "predictive_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "cleaned_iot_dataset.csv")

def train_predictive_model():
    """Train a predictive maintenance model with SMOTE and save it."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    X = df[['temperature', 'humidity', 'use [kW]', 'vibration']].fillna(df.median(numeric_only=True))
    y = df['failure']

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"Original failure ratio: {y.mean():.4f}")
    print(f"After SMOTE failure ratio: {y_resampled.mean():.4f}")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_resampled, y_resampled)

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model trained and saved at {MODEL_PATH}")

def load_model():
    """Load the trained model, train if not available."""
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Model not found. Training a new one...")
        train_predictive_model()
    return joblib.load(MODEL_PATH)

def predict_failure(sensor_data):
    """Predict failure probability for given sensor readings."""
    model = load_model()
    df = pd.DataFrame([sensor_data])
    prob = model.predict_proba(df)[0][1]
    return prob

if __name__ == "__main__":
    # Train model when run locally
    train_predictive_model()

    # Example usage (local testing only)
    sample_data = {
        "temperature": 70,
        "humidity": 40,
        "use [kW]": 5,
        "vibration": 0.02
    }
    probability = predict_failure(sample_data)
    print(f"📊 Failure Probability: {probability:.2%}")
