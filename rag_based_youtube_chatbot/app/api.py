# app/api.py
from fastapi import FastAPI
from pydantic import BaseModel

from .ingest import index_youtube_video
from .rag_chain import build_rag_chain
from langchain_community.document_loaders.youtube import YoutubeLoader


app = FastAPI(title="YouTube RAG Chatbot API")


class ChatRequest(BaseModel):
    video_url: str
    question: str


@app.post("/chat")
def chat(req: ChatRequest):
    video_id = YoutubeLoader.extract_video_id(req.video_url)
    collection_name = f"yt_{video_id}"

    # In a real system you'd:
    # - cache whether you've already indexed this collection_name
    # - avoid re-indexing on every call
    index_youtube_video(req.video_url, collection_name)

    chain = build_rag_chain(collection_name)
    result = chain.invoke({"input": req.question})
    answer = result.get("answer") or str(result)

    return {
        "video_id": video_id,
        "answer": answer,
    }
