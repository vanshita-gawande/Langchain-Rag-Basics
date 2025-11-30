from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Load token from .env
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Create Hugging Face client
model = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HF_TOKEN
)

# ✅ Step 1: Generate Detailed Report
prompt1 = "Write a detailed report on India."

report_response = model.chat_completion(
    messages=[
        {"role": "user", "content": prompt1}
    ],
    max_tokens=400,
    temperature=0.4
)

report_text = report_response.choices[0].message["content"]
print("\n✅ DETAILED REPORT:\n")
print(report_text)

# ✅ Step 2: Generate Summary from the Report
prompt2 = f"Write a five line summary on this:\n{report_text}"

summary_response = model.chat_completion(
    messages=[
        {"role": "user", "content": prompt2}
    ],
    max_tokens=150,
    temperature=0.4
)

summary_text = summary_response.choices[0].message["content"]
print("\n✅ FINAL SUMMARY:\n")
print(summary_text)
