#to cobvert text to vector to unserstand the contextual meaning
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model = "pass model",dimensions=20)

result = embedding.embed_query("Delhi is capital of india")#embedding will go to model and query is processed and get the vector response
print(result) #give 20 dimension vector that represent the conceptual meaning of query more vector dimension more conectual and less
