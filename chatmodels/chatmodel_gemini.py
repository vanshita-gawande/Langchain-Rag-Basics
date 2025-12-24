from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv() #we get our api key here from env file

model = ChatGoogleGenerativeAI(model = "gemini")
result = model.invoke("pass here question")

print(result)
