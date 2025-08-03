# anomaly_detection.py

def detect_anomalies(sensor_data):
    """
    Check sensor readings for anomalies based on thresholds.
    Returns a dictionary of alerts if anomalies found.
    """
    anomalies = {}

    if sensor_data['temperature'] > 28:
        anomalies['Temperature'] = "High temperature detected"

    if sensor_data['humidity'] < 35:
        anomalies['Humidity'] = "Low humidity detected"

    if sensor_data['use [kW]'] > 4.5:
        anomalies['Energy'] = "High energy usage detected"

    if sensor_data['vibration'] > 0.8:
        anomalies['Vibration'] = "Possible mechanical fault"

    return anomalies
