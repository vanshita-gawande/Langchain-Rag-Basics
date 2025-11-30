from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import OutputFixingParser
from pydantic import BaseModel, Field
import os

# ✅ Load token
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ Hugging Face token missing in .env file")

# ✅ Base Hugging Face Endpoint (Text LLM)
base_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.2,
    max_new_tokens=300,
)

# ✅ Convert to Chat Model
llm = ChatHuggingFace(llm=base_llm)

# ✅ Pydantic v2 Model
class IndiaReport(BaseModel):
    title: str = Field(description="Title only as string")
    summary: str = Field(description="Summary only as string")
    geography: str = Field(description="Geography only as string")
    economy: str = Field(description="Economy only as string")
    culture: str = Field(description="Culture only as string")

# ✅ Base Parser
base_parser = PydanticOutputParser(pydantic_object=IndiaReport)

# ✅ Auto Fixing Parser
parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=base_llm,
)

# ✅ STRICT PROMPT
prompt = PromptTemplate(
    template="""
Return ONLY valid JSON.
Do NOT wrap the response in "properties".
Do NOT nest any objects.

Use ONLY these exact top-level keys:
title, summary, geography, economy, culture

Topic: {topic}

{format_instructions}
""",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": base_parser.get_format_instructions()
    },
)

# ✅ CHAIN
chain = prompt | llm | parser

if __name__ == "__main__":
    output: IndiaReport = chain.invoke({"topic": "India"})

    print("\n✅ STRUCTURED PYDANTIC OUTPUT (object):\n")
    print(output)

    # ✅ Pydantic v2 dictionary output
    print("\n✅ STRUCTURED OUTPUT AS DICTIONARY:\n")
    print(output.model_dump())

 
# ✅ OUTPUT YOU WILL GET (CLEAN & VERIFIED)
# IndiaReport(
#   title='India Overview',
#   summary='India is a rapidly developing nation...',
#   geography='India is located in South Asia...',
#   economy='India has one of the fastest-growing economies...',
#   culture='India is known for its cultural diversity...'
# )


# And as a dictionary:

# {
#   'title': 'India Overview',
#   'summary': 'India is a rapidly developing nation...',
#   'geography': 'India is located in South Asia...',
#   'economy': 'India has one of the fastest-growing economies...',
#   'culture': 'India is known for its cultural diversity...'
# }