#to cobvert text to vector to unserstand the contextual meaning
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model = "pass model",dimensions=20)
# for multiple queries

documents = [
    "quer1",
    "query2",
    'query3'
]

result = embedding.embed_documents(documents)#embedding will go to model and query is processed and get the vector response
print(str(result)) #give 20 dimension vector that represent the conceptual meaning of query more vector dimension more conectual and less
