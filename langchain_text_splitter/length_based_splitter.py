from dotenv import load_dotenv
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

#
# ✅ Load Hugging Face token
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

if not hf_token:
    raise ValueError("Token not found in .env file")

# ✅ 1️⃣ SAMPLE BIG TEXT (Example Input)
text = """
Artificial Intelligence is transforming the world. It helps in healthcare, education,
finance, and many other industries. Machine Learning is a subset of AI. Deep Learning
is a subset of Machine Learning. AI systems learn from data and improve automatically.
"""

splitter = CharacterTextSplitter(
    separator ="\n",
    chunk_size = 100,#each chunk will have 100 characters
    chunk_overlap = 20,#next chunk will reuse last 20 characters
)
#split the text
chunks = splitter.split_text(text)