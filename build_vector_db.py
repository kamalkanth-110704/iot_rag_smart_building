# build_vector_db.py

import os
import chromadb
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load all manual text files from "manuals" folder
    docs = []
    for filename in os.listdir("manuals"):
        if filename.endswith(".txt"):
            with open(f"manuals/{filename}", "r", encoding="utf-8") as f:
                docs.append(f.read())

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []
    for doc in docs:
        chunks = splitter.split_text(doc)
        all_chunks.extend(chunks)

    # Create Chroma vector DB
    client = chromadb.Client()
    collection = client.create_collection(name="manuals")

    for i, chunk in enumerate(all_chunks):
        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embeddings.embed_query(chunk)]
        )

    return collection
