#Prompt → LLM → Parser

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
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Explain {topic} in simple words")
])

# ✅ String Output Parser
parser = StrOutputParser()

# ✅ Chain
chain = prompt | llm | parser

#chain.get_graph().print_ascii() #It prints a text-based (ASCII) diagram of your chain flow.

# ✅ Run
result = chain.invoke({"topic": "India"})
print("\n✅ AI Output:\n")
print(result)

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
