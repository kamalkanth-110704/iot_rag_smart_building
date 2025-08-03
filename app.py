import streamlit as st
from rag_system import query_system, ingest_documents
from predictive_maintenance import load_model

# -------------------------
# Initialize App
# -------------------------
st.set_page_config(page_title="Predictive Maintenance RAG System", layout="wide")

st.title("🔧 Predictive Maintenance with RAG & AI")
st.markdown("Ask maintenance-related questions and get AI-powered failure predictions.")

# -------------------------
# Load Knowledge Base
# -------------------------
with st.spinner("📚 Loading documents into vector database..."):
    ingest_ok = ingest_documents()

if not ingest_ok:
    st.error("❌ Missing required files: `maintenance_manual.txt` or `building_specs.txt`.\n"
             "Please add them to the project folder.")
    st.stop()

# -------------------------
# Load Model
# -------------------------
with st.spinner("⚙️ Loading predictive maintenance model..."):
    load_model()

# -------------------------
# User Inputs
# -------------------------
st.subheader("📝 Ask Your Question")
query = st.text_input("Enter your maintenance question:", placeholder="e.g., How to prevent overheating in the motor?")

st.subheader("📡 Enter Sensor Data")
col1, col2, col3, col4 = st.columns(4)

with col1:
    temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=200.0, value=70.0, step=0.5)
with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=40.0, step=0.5)
with col3:
    use_kw = st.number_input("Usage (kW)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
with col4:
    vibration = st.number_input("Vibration (mm/s)", min_value=0.0, max_value=5.0, value=0.02, step=0.01)

# -------------------------
# Run Query
# -------------------------
if st.button("🔍 Get Prediction"):
    if not query.strip():
        st.warning("⚠️ Please enter a maintenance question.")
    else:
        with st.spinner("🤖 Analyzing..."):
            sensor_data = {
                "temperature": temperature,
                "humidity": humidity,
                "use [kW]": use_kw,
                "vibration": vibration
            }

            response = query_system(query, sensor_data)

        st.success("✅ Analysis Complete")
        st.markdown("### 💡 System Response")
        st.write(response)
