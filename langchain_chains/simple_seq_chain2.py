# Prompt1 → LLM → Prompt2 → LLM → Parser

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# ✅ Load token
load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not hf_token:
    raise ValueError("❌ Token not found in .env file")

# ✅ ✅ Use CHAT model (THIS FIXES ALL YOUR ERRORS)
base_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # ✅ This model ONLY works as chat
    huggingfacehub_api_token=hf_token,
    temperature=0.5,
    max_new_tokens=200,
)

# ✅ Convert to Chat Model
llm = ChatHuggingFace(llm=base_llm)

# ✅ Chat Prompt , this had fixed our code this is import to pass role in prompt as this model only works with chat so using chat templete istaed of prompttemplete
prompt_1 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Explain {topic} in simple words")
])

# ✅ ✅ PROMPT 2 → Expands the summary into a full explanation
prompt_2 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful teacher."),
    ("human", "Using this summary, explain in simple words:\n\n{summary}")
])

# ✅ String Output Parser
parser = StrOutputParser()

# ✅ Chain -- can also keep like below but could not diff op as LangChain treats this as one continuous pipeline, so:The output of step-1 is hidden,Only the final output (step-2) is returned
# chain = prompt_1 | llm | parser | prompt_2 | llm | parser
#so use this to see both op

# ✅ PROMPT 1 → Explanation
prompt_1 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Explain {topic} in simple words")
])

# ✅ PROMPT 2 → Refine / Rewrite using summary
prompt_2 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful teacher."),
    ("human", "Using this explanation, rewrite it more clearly:\n\n{summary}")
])

# chain.get_graph().print_ascii() #It prints a text-based (ASCII) diagram of your chain flow.

# ✅ CHAIN 1
chain_1 = prompt_1 | llm | parser

# ✅ CHAIN 2
chain_2 = prompt_2 | llm | parser

# ✅ Run
step1_output = chain_1.invoke({"topic": "India"})
print("\n✅ STEP 1 OUTPUT (FROM PROMPT 1):\n")
print(step1_output)

step2_output = chain_2.invoke({"summary": step1_output})
print("\n✅ STEP 2 OUTPUT (FROM PROMPT 2 - FINAL OUTPUT):\n")
print(step2_output)


#      +-------------+       
#      | PromptInput |       
#      +-------------+       
#             *
#             *
#             *
#   +--------------------+   
#   | ChatPromptTemplate |   
#   +--------------------+   
#             *
#             *
#             *
#    +-----------------+     
#    | ChatHuggingFace |     
#    +-----------------+     
#             *
#             *
#             *
#    +-----------------+
#    | StrOutputParser |
#    +-----------------+
#             *
#             *
#             *
# +-----------------------+
# | StrOutputParserOutput |
# +-----------------------+
#             *
#             *
#             *
#   +--------------------+
#   | ChatPromptTemplate |
#   +--------------------+
#             *
#             *
#             *
#    +-----------------+
#    | ChatHuggingFace |
#    +-----------------+
#             *
#             *
#             *
#    +-----------------+
#    | StrOutputParser |
#    +-----------------+
#             *
#             *
#             *
# +-----------------------+
# | StrOutputParserOutput |
# +-----------------------+
