# User Topic
#    ↓
# Lambda checks topic
#    ↓
# IF topic == "AI"      → Explanation chain
# ELSE (anything else) → Facts chain

from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

# ✅ Load token from .env
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

if not hf_token:
    raise ValueError("Token not found")

# ✅ Create Hugging Face model
base_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=hf_token,
    temperature=0.4,
    max_new_tokens=200,
)

# Convert to Chat Model
llm = ChatHuggingFace(llm=base_llm)

# Prompts
prompt_explain = ChatPromptTemplate.from_messages([
    ("system", "you are a teacher"),
    ("human", "explain the {topic}")
])

prompt_facts = ChatPromptTemplate.from_messages([
    ("system", "you are helpful"),
    ("human", "give 3 important facts about {topic}")
])

# Output Parser
parser = StrOutputParser()

# Chains
explain_chain = prompt_explain | llm | parser
facts_chain   = prompt_facts   | llm | parser

#using lambad to extract only topic value
topic_extractor = RunnableLambda(lambda x: x["topic"])

#runnable branch if else logic
branch_chain = RunnableBranch(
    (lambda x: x == "AI",explain_chain), #if topic only ai then explain else facts chain
    facts_chain
)

#final_chain 
final_chain = topic_extractor | branch_chain

#run the chain
result = final_chain.invoke({"topic":"AI"})

print("\n Final op \n:")
print(result)