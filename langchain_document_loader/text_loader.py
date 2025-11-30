# from langchain_community.document_loaders import TextLoader

# loader = TextLoader('cricket.txt',encoding='utf-8')

# docs = loader.load()

# print(docs)
# print(type(docs))#list 
# print(docs[0].page_content)#give the whole page ocntent
# print(docs[0].metadata)#give cricket.txt as it is our source



# build a simple chain with this text loader
# Loads your cricket.txt file,Sends the content to the AI,Gets a simple summary,Prints the result

from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#  Load Hugging Face token
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

if not hf_token:
    raise ValueError("❌ Token not found in .env file")

#  1️⃣ LOAD TEXT FILE USING TextLoader
loader = TextLoader("cricket.txt")
documents = loader.load()

#  Extract only the text from the document
text_data = documents[0].page_content

print("\n FILE LOADED SUCCESSFULLY!\n")
print(text_data)

# 2️⃣ CREATE HUGGING FACE MODEL
base_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=hf_token,
    temperature=0.3,
    max_new_tokens=200,
)

llm = ChatHuggingFace(llm=base_llm)

# 3️⃣ CREATE SIMPLE PROMPT
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "Summarize this text in simple words:\n\n{text}")
])

# 4️⃣ OUTPUT PARSER
parser = StrOutputParser()

# 5️⃣ SIMPLE CHAIN (Prompt → LLM → Parser)
chain = prompt | llm | parser

# 6️⃣ RUN THE CHAIN WITH FILE CONTENT
result = chain.invoke({"text": text_data})

print("\n AI SUMMARY OUTPUT:\n")
print(result)

