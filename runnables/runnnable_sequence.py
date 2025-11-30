# User Topic
#    ↓
# Prompt 1 → Explain
#    ↓
# Prompt 2 → Improve / Rewrite
#    ↓
# Final Answer to User

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not hf_token:
    raise ValueError("Token not found")

#base chat model

base_llm = HuggingFaceEndpoint(
     repo_id="mistralai/Mistral-7B-Instruct-v0.2",
     huggingfacehub_api_token=hf_token,
     temperature=0.4,
     max_new_tokens=200,
)

llm = ChatHuggingFace(llm = base_llm)

#in both prompt input variable name must same
#PROMPT 1 → Basic Explanation
prompt1 = ChatPromptTemplate.from_messages([
    ("system","You are helpful teacher"),
    ("human","Explain {topic} in simple words")
])

#ROMPT 2 → Improve the Explanation
prompt2 = ChatPromptTemplate.from_messages([
    ("system","you are writing expert"),
    ("human","Rewrite this explanation in a cleaner way\n\n{topic}")
])

#parser
parser = StrOutputParser()

#RUNNABLE SEQUENCE (TWO PROMPTS → TWO LLM CALLS)

sequence = RunnableSequence(prompt1,llm,parser,prompt2,llm,parser)

#run sequence

result = sequence.invoke({'topic':"India"})

print("\n✅ FINAL OUTPUT FROM RUNNABLE SEQUENCE:\n")
print(result)