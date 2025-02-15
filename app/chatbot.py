from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from app.vector_store import load_vector_store
import config

vector_store = load_vector_store()
retriever = vector_store.as_retriever()

# Use a valid Hugging Face model for text generation
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=config.HUGGINGFACEHUB_API_TOKEN
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def get_response(user_query):
    return qa_chain.run(user_query)
