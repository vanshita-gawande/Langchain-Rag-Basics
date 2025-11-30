        #         ┌──────────────┐
        #         │  RESEARCH    │
        #         │   PAPER      │
        #         └──────┬───────┘
        #                │
        #      ┌─────────┴─────────┐
        #      │                   │
        # PROMPT 1             PROMPT 2
        # (NOTES)               (QUIZ)
        #      │                   │
        #    MODEL 1             MODEL 2
        #      │                   │
        #    NOTES               QUIZ
        #           └──────┬──────┘
        #                  │
        #              PROMPT 3
        #              (MERGE)
        #                  │
        #               MODEL 3
        #                  │
        #           ✅ FINAL LEARNING PACK

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os

#load env
load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not hf_token:
    raise ValueError("❌ Token not found in .env file")

#base chat model(used 3 times)
base_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=hf_token,
    temperature=0.4,
    max_new_tokens=300,
)

llm = ChatHuggingFace(llm = base_llm)

#pass input
linear_regression_paper = """
Linear Regression is a supervised learning algorithm used to model the relationship
between a dependent variable and one or more independent variables. The relationship
is assumed to be linear and is represented using a mathematical equation. It is widely
used in prediction, forecasting, and trend analysis.
"""

#prompt1 -> gen study notes
#from_messages() is used to create a CHAT-style prompt using roles like:system → sets behavior,human → user input,ai → assistant messages (optional),It tells the model:“This is a conversation, not just plain text.”
notes_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert ML teacher."),
    ("human", "Create detailed study notes from the following paper:\n\n{paper}")
])

# ✅ ✅ PROMPT 2 → GENERATE QUIZ
quiz_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an exam question setter."),
    ("human", "Create 5 quiz questions from the following paper:\n\n{paper}")
])

# ✅ ✅ PROMPT 3 → MERGE NOTES + QUIZ
merge_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional course designer."),
    ("human", "Merge the following NOTES and QUIZ into a clean learning handout.\n\nNOTES:\n{notes}\n\nQUIZ:\n{quiz}")
])

#output parser
parser = StrOutputParser()

#model call 1 -> notes chain
notes_chain = notes_prompt | llm | parser

#model call 2 -> quiz chain
quiz_chain = quiz_prompt | llm | parser

# PARALLEL EXECUTION (NOTES + QUIZ TOGETHER) “Run both LLM chains at the SAME time using the same input, and give me both outputs together.”
# So instead of: First generate notes , Then generate quiz , One after another (slow)Notes + Quiz simultaneously (faster)
# 2️ What Would Happen WITHOUT RunnableParallel?
# You would have to do this manually:
# notes = notes_chain.invoke({"paper": paper})
# quiz = quiz_chain.invoke({"paper": paper})

parallel_chain = RunnableParallel(
    notes = notes_chain,
    quiz = quiz_chain  #passes here both chains
)

#model call 3 - merge chain
merge_chain = merge_prompt | llm | parser

# ✅ STEP 1 → Generate Notes & Quiz in Parallel
parallel_output = parallel_chain.invoke({'paper':linear_regression_paper})
notes = parallel_output['notes']
quiz = parallel_output['quiz']
print("\n✅ GENERATED NOTES:\n")
print(notes)

print("\n✅ GENERATED QUIZ:\n")
print(quiz)

#step 2 - merge notes + quiz using 3rd model

final_output = merge_chain.invoke({
    "notes":notes,
    "quiz" : quiz
})

print("\n✅ ✅ FINAL MERGED LEARNING PACK (USER OUTPUT):\n")
print(final_output)

#it fisrt give topic then quiz and then both 