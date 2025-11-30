# # ✅ Always return clean plain text
# # ✅ Remove extra quotes, markdown, code blocks

# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# import os

# # ✅ Load API key
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # ✅ Create OpenAI Chat Model
# model = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0.4
# )

# # ✅ String Output Parser (built-in)
# string_parser = StrOutputParser()

# # ✅ Prompt 1: Detailed Report
# prompt1 = PromptTemplate.from_template(
#     "Write a detailed report on {topic}."
# )

# # ✅ Prompt 2: Summary
# prompt2 = PromptTemplate.from_template(
#     "Write a five line summary on this:\n{text}"
# )

# # ✅ ✅ CHAIN 1: Topic → Report
# report_chain = prompt1 | model | string_parser

# # ✅ ✅ CHAIN 2: Report → Summary
# summary_chain = prompt2 | model | string_parser

# # ✅ ✅ RUN CHAIN 1
# report_text = report_chain.invoke({"topic": "India"})

# print("\n✅ DETAILED REPORT:\n")
# print(report_text)

# # ✅ ✅ RUN CHAIN 2
# summary_text = summary_chain.invoke({"text": report_text})

# print("\n✅ FINAL SUMMARY:\n")
# print(summary_text)
  

# #   this is using open ai using chians 

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Load token
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not HF_TOKEN:
    raise ValueError("Hugging Face token not found in .env")

# Base endpoint (NO task)
base_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.4,
    max_new_tokens=300
)

# Convert to Chat Model ✅
llm = ChatHuggingFace(llm=base_llm)

# Output parser
string_parser = StrOutputParser()

# Prompt 1
prompt1 = PromptTemplate.from_template(
    "Write a detailed report on {topic}."
)

# Prompt 2
prompt2 = PromptTemplate.from_template(
    "Write a five line summary on this:\n{text}"
)

# Chains
report_chain = prompt1 | llm | string_parser
summary_chain = prompt2 | llm | string_parser

# Run Chain 1
report_text = report_chain.invoke({"topic": "India"})
print("\n✅ DETAILED REPORT:\n")
print(report_text)

# Run Chain 2
summary_text = summary_chain.invoke({"text": report_text})
print("\n✅ FINAL SUMMARY:\n")
print(summary_text)


# The main difference between StrOutputParser and StructuredOutputParser is that StrOutputParser returns the model’s response as plain, free-form text without enforcing any format, while StructuredOutputParser forces the output to follow a strict predefined structure such as JSON or a dictionary. With StrOutputParser, the model has full freedom in how it writes—this is why your report and summary appear like normal paragraphs and may not strictly follow rules like “exactly five lines.” In contrast, StructuredOutputParser is used when you need reliable, machine-readable output with fixed fields (for example, title, summary, key points), making it more suitable for APIs, databases, and automation systems. In simple terms, StrOutputParser is best for human-readable creative text, while StructuredOutputParser is best for controlled, production-ready data.