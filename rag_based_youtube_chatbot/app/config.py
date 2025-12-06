#app/config.py – central config & model selection
# app/config.py
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face token (free tier)
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    raise RuntimeError(
        "HUGGINGFACEHUB_API_TOKEN not set in .env. "
        "Create one on Hugging Face (free) and add it."
    )

# Embedding model (small & fast, local)
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chat model via HF Inference (free tier)
HF_CHAT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
# You can swap to a smaller one if your account doesn’t have access:
# HF_CHAT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Optional: YouTube API key if you want to experiment with official Data API
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

print(f"Configuration loaded. Using chat model: {HF_CHAT_MODEL}")