@"
# 🏢 IoT RAG Predictive Maintenance System

This application combines IoT sensor data with a Retrieval-Augmented Generation (RAG) system to:
- Answer maintenance-related queries
- Predict equipment failure probability

## Features
- **Document Retrieval**: Searches `maintenance_manual.txt` & `building_specs.txt`
- **Predictive Maintenance**: Uses a trained ML model
- **Streamlit UI**: User-friendly interface

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
