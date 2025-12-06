# (SMART SPLITTING – USED IN REAL RAG)
# Concept :
# Splits text using meaning instead of size
# It keeps related sentences together using embeddings.
# This is the BEST for RAG accuracy 

#need to create embedding using the embedding model 

from dotenv import load_dotenv
import os

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# Load token
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

text = """
Artificial Intelligence is used in healthcare for disease prediction.
It helps doctors detect cancer early.

Machine learning models are trained on huge datasets.
They improve their accuracy over time.

India has one of the fastest growing AI ecosystems.
Startups are innovating rapidly.
"""
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
)

#semantic splitter 
splitter = SemanticChunker(embeddings)

chunks = splitter.split_text(text)

print(chunks)