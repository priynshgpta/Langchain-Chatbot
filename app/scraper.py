from langchain_community.document_loaders import WebBaseLoader
import config

def extract_data():
    loader = WebBaseLoader(config.BASE_URL)
    documents = loader.load()
    return documents
