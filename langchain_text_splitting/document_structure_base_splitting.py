# 3 -
# (PDF, Word, Webpage, CSV etc.)
# Splitting based on real document layout:Pages,Headings,Sections,Tables

# Used with:
# PyPDFLoader,DirectoryLoader,WebBaseLoader

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# DEBUG: CONFIRM FILE EXISTS
file_path = "sample.txt"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File NOT found: {file_path}. Create it in this folder.")

# LOAD DOCUMENT
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

print("\n DOCUMENT LOADED SUCCESSFULLY\n")

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 40
)

docs = splitter.split_documents(documents)
print(docs)


# When to Use:
# ✔ PDF → RAG
# ✔ Notes → RAG
# ✔ CSV → RAG
# ✔ Website → RAG
