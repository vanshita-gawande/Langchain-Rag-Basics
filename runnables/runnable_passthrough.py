#It passes the input forward unchanged, so you can reuse the original input along with parallel outputs.
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

if not hf_token:
    raise ValueError("not found")

# ✅ Create model
base_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",  
    huggingfacehub_api_token=hf_token,
    temperature=0.4,
    max_new_tokens=200,
)

# ✅ Convert to chat model
llm = ChatHuggingFace(llm=base_llm)

# ✅ Prompts
prompt_explain = ChatPromptTemplate.from_messages([
    ("system", "hey"),
    ("human", "explain the {topic}")
])

prompt_facts = ChatPromptTemplate.from_messages([
    ("system", "you are helpful"),
    ("human", "give 3 important facts about {topic}")
])

# Parser
parser = StrOutputParser()

# Chains
explain_chain = prompt_explain | llm | parser
facts_chain   = prompt_facts   | llm | parser

# PARALLEL + PASSTHROUGH (THIS IS THE MAIN CHANGE)

parallel_chian = RunnableParallel({
    "topic":RunnablePassthrough(), #keep the original text as
    "explanation":explain_chain,
    "facts":facts_chain
})

#invoke parallel chian

result = parallel_chian.invoke({"topic":'Ai'})

print("\nORIGINAL INPUT:\n", result["topic"]) #print our topic as it is
print("\nInner Value",result["topic"]["topic"]) #to print the inner value
print("\nEXPLANATION:\n", result["explanation"])
print("\nFACTS:\n", result["facts"])