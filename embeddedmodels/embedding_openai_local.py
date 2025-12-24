#local open source model
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
text = "delhi is capital of india"

vector = embedding.embed_query(text)

print(str(vector))
#here also can generate embedding fo rmultiple queries
