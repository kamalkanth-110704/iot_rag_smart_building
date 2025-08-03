# rag_system.py

import os
from langchain_community.vectorstores import Chroma  # ✅ Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ Updated import
from langchain.schema import Document
from predictive_maintenance import predict_failure

# 1️⃣ Load Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2️⃣ Create / Load Chroma DB
CHROMA_DB_DIR = "chroma_db"
if not os.path.exists(CHROMA_DB_DIR):
    os.makedirs(CHROMA_DB_DIR)

vector_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_model)

# 3️⃣ Load Maintenance Manual & Specs
def ingest_documents():
    docs = []
    missing_files = []

    for file in ["maintenance_manual.txt", "building_specs.txt"]:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                print(f"⚠️ File '{file}' is empty. Please add content.")
                return False
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_text(text)
            docs.extend([Document(page_content=chunk, metadata={"source": file}) for chunk in chunks])
        else:
            missing_files.append(file)

    if missing_files:
        print("❌ Missing required files:")
        for mf in missing_files:
            print(f"   - {mf}")
        print("\n📌 Please create these files in the project folder before running again.")
        return False

    if docs:
        vector_db.add_documents(docs)
        print(f"✅ Ingested {len(docs)} document chunks into Chroma DB")
        return True
    else:
        print("⚠️ No valid documents found to ingest.")
        return False

# 4️⃣ Retrieve Relevant Context
def retrieve_context(query):
    results = vector_db.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in results])

# 5️⃣ Combined Query + Predictive Maintenance
def query_system(query, sensor_data):
    context = retrieve_context(query)
    failure_probability = predict_failure(sensor_data)

    response = (
        f"📊 Failure Probability: {failure_probability*100:.2f}%\n"
        f"📚 Relevant Maintenance Info:\n{context}"
    )
    return response

# 6️⃣ Main Entry
if __name__ == "__main__":
    if not ingest_documents():
        exit(1)

    # 🆕 Take live input
    query = input("\nEnter your maintenance question: ")

    try:
        temperature = float(input("Temperature: "))
        humidity = float(input("Humidity: "))
        use_kw = float(input("Use [kW]: "))
        vibration = float(input("Vibration: "))
    except ValueError:
        print("❌ Please enter valid numeric values for sensor readings.")
        exit(1)

    sensor_data = {
        "temperature": temperature,
        "humidity": humidity,
        "use [kW]": use_kw,
        "vibration": vibration
    }

    answer = query_system(query, sensor_data)
    print("\n💡 System Response:\n")
    print(answer)
