# from dotenv import load_dotenv
# from typing import TypedDict
# from langchain_openai import ChatOpenAI

# # load env variables
# load_dotenv()

# # initialize OpenAI model
# model = ChatOpenAI()

# # ✅ schema
# class Review(TypedDict):
#     summary: str
#     sentiment: str

# # ✅ enable structured output
# struct_model = model.with_structured_output(Review)

# # ✅ invoke model
# result = struct_model.invoke(
#     "The iPhone is known for its premium build quality, smooth performance, "
#     "strong security, and excellent camera system. Apple designs both the "
#     "hardware and software (iOS), which results in a very stable, fast, and "
#     "user-friendly experience."
# )

# # ✅ final validated output
# print(result)
# #  the above code when working with open ai model it geneerate the prompt under the hood 

#working with hugging face
from dotenv import load_dotenv
from typing import TypedDict
from huggingface_hub import InferenceClient
import os
import json

# load env
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# ✅ initialize HF client
model = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",  # works well for JSON
    token=HF_TOKEN
)

# ✅ schema
class Review(TypedDict):
    summary: str
    sentiment: str

# ✅ structured prompt
prompt = """
Analyze the following product review and return ONLY valid JSON with exactly this schema:

{
  "summary": "string",
  "sentiment": "positive | negative | neutral"
}

Review:
The iPhone is known for its premium build quality, smooth performance, strong security, and excellent camera system. Apple designs both the hardware and software (iOS), which results in a very stable, fast, and user-friendly experience.
"""

# ✅ call model
response = model.chat_completion(
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=300,
    temperature=0.2
)

print("RAW OUTPUT:\n", response)

# ✅ extract JSON safely
json_start = response.find("{")
json_end = response.rfind("}") + 1
json_data = response[json_start:json_end]

# ✅ convert to TypedDict
result: Review = json.loads(json_data)

print("\n✅ STRUCTURED OUTPUT:")
print(result)
