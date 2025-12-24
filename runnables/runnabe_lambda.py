from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
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

# ✅ Convert to Chat Model
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

# ✅ Output Parser
parser = StrOutputParser()

# ✅ Chains
explain_chain = prompt_explain | llm | parser
facts_chain   = prompt_facts   | llm | parser

#parallel + lambda
parallel_chain = RunnableParallel({
    "topic":RunnableLambda(lambda x: x["topic"]), #rteurn only ai in terminal
    "explanation":explain_chain,
    "facts":facts_chain
})

result = parallel_chain.invoke({'topic':'AI'})
#run the chain
# ✅ Print results
print("\nORIGINAL TOPIC:\n", result["topic"])        # ✅ AI only
print("\nEXPLANATION:\n", result["explanation"])
print("\nFACTS:\n", result["facts"])
#in previous wrote extra code for print only inner topic value but here we use lambada function and pass condition to it so that it return only the topic value
