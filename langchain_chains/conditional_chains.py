# User Feedback
#       ↓
# Sentiment Classifier (LLM + Pydantic)
#       ↓
# RunnableBranch (if/else)
#       ↓
# Positive → Thank You Response
# Negative → Apology + Support Response
# Neutral  → Acknowledge Response

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
import os

# ==========================================================
# ✅ LOAD ENV
# ==========================================================

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not hf_token:
    raise ValueError("❌ Token not found")

# ==========================================================
# ✅ BASE CHAT MODEL
# ==========================================================

base_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=hf_token,
    temperature=0.3,
    max_new_tokens=200,
)

llm = ChatHuggingFace(llm=base_llm)

# STRICT PYDANTIC SCHEMA

class FeedbackSentiment(BaseModel):
    sentiment: str = Field(description="Must be one of: positive, negative, neutral")
    confidence: float = Field(description="Confidence score between 0 and 1")

sentiment_parser = PydanticOutputParser(pydantic_object=FeedbackSentiment)

# SENTIMENT PROMPT (NO partial_variables)

sentiment_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sentiment analysis engine. Return only valid JSON."),
    ("human", """
Analyze the sentiment of this feedback:

Feedback: {feedback}

{format_instructions}
""")
])

sentiment_chain = sentiment_prompt | llm | sentiment_parser

# RESPONSE PROMPTS
positive_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a polite support agent."),
    ("human", "User is happy. Reply with a thank you note.")
])

negative_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a customer support manager."),
    ("human", "User is unhappy. Apologize and promise to help.")
])

neutral_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a neutral support assistant."),
    ("human", "User gave neutral feedback. Respond professionally.")
])

parser = StrOutputParser()

positive_chain = positive_prompt | llm | parser
negative_chain = negative_prompt | llm | parser
neutral_chain  = neutral_prompt  | llm | parser

#  CONVERT PYDANTIC → DICT FOR ROUTER
to_dict = RunnableLambda(lambda x: {
    "sentiment": x.sentiment,
    "confidence": x.confidence
})

#  CONDITIONAL ROUTER
router = RunnableBranch(
    (lambda x: x["sentiment"] == "positive", positive_chain),
    (lambda x: x["sentiment"] == "negative", negative_chain),
    neutral_chain
)
# FINAL PIPELINE

final_chain = sentiment_chain | to_dict | router

#  RUN SYSTEM (format_instructions PASSED MANUALLY)

user_feedback = "The course was good but the explanations were a bit confusing."

result = final_chain.invoke({
    "feedback": user_feedback,
    "format_instructions": sentiment_parser.get_format_instructions()
})

print("\n✅ USER FEEDBACK:\n", user_feedback)
print("\n✅ FINAL AI RESPONSE:\n", result)
