#2 SECOND ONE 
# (Split by paragraphs, lines, dots, headers)
# ✅ Concept
# Splits text using structure like \n, ., ##, etc.
# Example:
# New line = new chunk
# Paragraph = new chunk

from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
## Introduction
Artificial Intelligence is everywhere.

## Machine Learning
Machine Learning is a subset of AI.

## Deep Learning
Deep Learning is a subset of Machine Learning.
"""

splitter = RecursiveCharacterTextSplitter(
    separators=["\n## ", "\n", ".", " "],  # structure-based
    chunk_size=120,
    chunk_overlap=10
)

chunks = splitter.split_text(text)
print(chunks) #here we use recurive splitter and pass the diff separator You got 2 clean chunks, split by document structure

# ✅ When to Use
# ✔ Blogs
# ✔ Markdown files
# ✔ Notes
# ✔ Web scraped text

