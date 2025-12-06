# 1. FIRST ONE SO SIMPLE
# (Fixed character size – MOST BASIC)
# Concept (Simple Words)
# Breaks text into equal-sized pieces based only on character count.
# Does NOT care about meaning
# Does NOT care about sentences
# Just cuts every N characters

from langchain_text_splitters import CharacterTextSplitter

text = """
Artificial Intelligence is transforming the world. It helps in healthcare, education,
finance and robotics. Machine Learning is a subset of AI. Deep Learning is a subset
of Machine Learning.Virat Kolh is my favorite crickter.
Artificial Intelligence is transforming the world. It helps in healthcare, education,
finance and robotics. Machine Learning is a subset of AI. Deep Learning is a subset
of Machine Learning. Virat Kohli is my favorite cricketer.
"""

splitter = CharacterTextSplitter(
    separator="",#on the basis of separator it sepaarate into 50 cagarsets then add , again separate 
    chunk_size = 50, # max characters per chunk
    chunk_overlap = 10 #repeat last 20 chars in next chunk
)

chunks = splitter.split_text(text)

print("\n Length Based Chunks:\n")
print(chunks)


# ✅ When to Use
# ✔ Quick testing
# ✔ Small files
# ❌ Not good for PDFs
# ❌ Not good for meaning-based search