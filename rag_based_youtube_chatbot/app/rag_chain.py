# Build the RAG chain from the vector store
# Now we need:
# Retriever from Chroma
# A chat model (Hugging Face) via HuggingFaceEndpoint + ChatHuggingFace 
# A “stuff” document chain
# A retrieval chain using create_retrieval_chain 
# app/rag_chain.py

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from .config import (
    CHROMA_DIR,
    HF_CHAT_MODEL,HUGGINGFACEHUB_API_TOKEN
)

def load_vectorstore(collection_name: str) -> Chroma:
    """ Load an existing persisted Chroma collection"""
    from .ingest import get_embeddings

    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR /collection_name,
        )
    )
    return vectorstore

def build_chat_llm() -> ChatHuggingFace:
    """
    Hugging Face hosted LLM (free tier).
    """
    base_llm = HuggingFaceEndpoint(
        repo_id=HF_CHAT_MODEL,
        task="text-generation",
        temperature=0.3,
        max_new_tokens=512,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )

    chat_model = ChatHuggingFace(
        llm=base_llm,   # ✅ FIXED HERE
        verbose=False
    )
    return chat_model


def build_rag_chain(collection_name: str):
    """
    Create a Retrieval-Augmented Generation chain using:
    - Chroma retriever
    - Hugging Face chat model
    - Modern LCEL pipeline (LangChain 0.2+ compatible)
    """
    vectorstore = load_vectorstore(collection_name)
    #build retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = build_chat_llm()
    system_prompt = """\
    You are a helpful assistant for a YouTube video.
    Answer the user's questions using ONLY the information from the video transcript.
    If you don't know the answer from the video, say so honestly.

    Use at most 5 sentences.
    Include timestamps or section hints from metadata when relevant.

    Context:
    {context}
    """
    #prompt containing system and human parts
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    # MODERN RAG CHAIN pass here retriever, prompt, llm
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    return rag_chain