# RunnableMap takes ONE input and runs it through MULTIPLE chains at the same time, then returns ALL results together in a dictionary.

# Think of it like:
# ONE input  →  MANY outputs  →  ONE combined result

# Real-Life Analogy (Very Easy)

# You give one topic: “AI” to:

# 📘 Teacher → writes explanation
# 📋 Researcher → gives 3 facts
# ❓ Examiner → creates questions

# All of them work at the same time, and you get:

# {
#   "explanation": "...",
#   "facts": "...",
#   "questions": "..."
# }
# This is exactly what RunnableMap does.

from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap

# ✅ Load Hugging Face token
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
    ("system", "You are a teacher"),
    ("human", "Explain {topic} in simple words")
])

prompt_facts = ChatPromptTemplate.from_messages([
    ("system", "You are a teacher"),
    ("human", "Give 3 important facts about {topic}")
])

# ✅ Output parser
parser = StrOutputParser()

# ✅ Chains
explain_chain = prompt_explain | llm | parser
facts_chain   = prompt_facts   | llm | parser

# ✅ ✅ RUNNABLE MAP (MAIN PART)
map_chain = RunnableMap({
    "explanation": explain_chain,
    "facts": facts_chain
})

# ✅ Run
result = map_chain.invoke({"topic": "AI"})

# ✅ Print results
print("\n✅ EXPLANATION:\n", result["explanation"])
print("\n✅ FACTS:\n", result["facts"])


#lRunnableSequence is used for step-by-step processing, whereas RunnableParallel and RunnableMap are used when the same input must be processed by multiple independent chains simultaneously. see notes

# | Same Input        | Goes To         |
# | ----------------- | --------------- |
# | `{"topic": "AI"}` | `explain_chain` |
# | `{"topic": "AI"}` | `facts_chain`   |
