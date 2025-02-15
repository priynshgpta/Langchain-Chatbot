from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import config
import os

# Define embeddings globally
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_store(documents):
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(config.VECTOR_STORE_PATH)

def load_vector_store():
    if os.path.exists(config.VECTOR_STORE_PATH):
        return FAISS.load_local(config.VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("❌ Vector store not found! Creating a new one...")
        from app.scraper import extract_data  # Import scraper function
        documents = extract_data()
        create_vector_store(documents)
        return FAISS.load_local(config.VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
