from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

if not hf_token:
    raise ValueError("not found")

#create model
base_llm = HuggingFaceEndpoint(
    repo_id ="mistralai/Mistral-7B-Instruct-v0.2",  
    huggingfacehub_api_token = hf_token,
    temperature=0.4,
    max_new_tokens=200,
)

#convert to chat model ChatHuggingFace is a wrapper that converts a normal text model into a CHAT-style model.

llm = ChatHuggingFace(llm = base_llm)

#prompt
prompt_explain = ChatPromptTemplate.from_messages([
    ('system',"hey"),
    ('human','explain the {topic}')
])

prompt_facts = ChatPromptTemplate.from_messages([
    ('system','you are helful'),
    ('human','give a 3 important about {topic}'),
])

#parser
parser = StrOutputParser()

#then using the prompt , model and parser create chain

explain_chain = prompt_explain | llm | parser
facts_chain = prompt_facts | llm |parser

#creating parallel chain 
paraller_chain = RunnableParallel({
    "explanation":explain_chain,# # result will be stored as result["explanation"]
    "facts":facts_chain # result will be stored as result["facts"]
})

result = paraller_chain.invoke({'topic':"Ai"})

print(result["explanation"])
print(result["facts"])

