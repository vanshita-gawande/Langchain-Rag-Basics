# Step 1 – Ingesting a YouTube transcript (indexing)

# We’ll:

# Use YoutubeLoader to fetch transcript. 
# api.python.langchain.com
# +1

# Split into chunks using RecursiveCharacterTextSplitter. 
# LangChain Docs

# Embed with HuggingFaceEmbeddings (local). 
# langchain-5e9cc07a.mintlify.app
# +1

# Store into Chroma with persistence. 
# reference.langchain.com

# app/ingest.py

from typing import List

from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from .config import CHROMA_DIR, HF_EMBEDDING_MODEL

def load_youtube_transcript(youtube_url: str) -> List[Document]:
    """Load transcript of a YouTube video as LangChain Documents.
    This uses youtube-transcript-api under the hood and does NOT require
    a YouTube Data API key.
    """
    loader = YoutubeLoader.from_youtube_url(
        youtube_url,
        # add_video_info = True,  # adds title, description, etc.
        language = ["en","en-auto"],       # transcript language
        translation = None,
    )
    docs = loader.load()
    if not docs:
        raise ValueError("No transcript found for the provided YouTube URL.")
    return docs

# Split documents into chunks
def split_documents(docs: List[Document]) -> List[Document]:
    """ Chunk documents into manageable pieces for RAG."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 150,
    )
    return splitter.split_documents(docs)

#get embeddings
def get_embeddings():
    """Local Hugging Face embeddings model."""
    return HuggingFaceEmbeddings(model_name = HF_EMBEDDING_MODEL) #can be changed to any other model

#build or update the vector store
def build_or_update_vectorestore(
        docs:List[Document],
        collection_name = str,
) -> Chroma:
    """ Create/update a chroma collection for this video"""
    embeddings = get_embeddings()

    vectorestore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR/collection_name),
    )
    # Add documents to the vectorestore(will append if collection alreday exits)
    vectorestore.add_documents(docs)
    vectorestore.persist()
    return vectorestore

#index a YouTube video
def index_youtube_video(youtube_url: str,collection_name: str) -> Chroma:
    """High-level function:
    - Load transcript
    - Split into chunks
    - Store in Chroma"""
    print(f" Loading transcript for {youtube_url}...")
    raw_docs = load_youtube_transcript(youtube_url)

    print(f" Splitting into chunks...")
    chunks = split_documents(raw_docs)

    vs = build_or_update_vectorestore(chunks,collection_name)
    return vs