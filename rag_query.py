# rag_query.py

import os
from openai import OpenAI
from build_vector_db import build_vector_store

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def rag_query(user_query):
    """Retrieve relevant manual sections and generate AI answer."""
    collection = build_vector_store()

    results = collection.query(
        query_texts=[user_query],
        n_results=3
    )

    context = "\n".join(results['documents'][0])
    prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"

    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content
