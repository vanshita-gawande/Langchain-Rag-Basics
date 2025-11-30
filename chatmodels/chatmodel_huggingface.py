# from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
# from dotenv import load_dotenv
# import os

# load_dotenv()

# HF_API_KEY = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
# print("HF:", HF_API_KEY)  # Debug

# llm = HuggingFaceEndpoint(
#     repo_id=  "meta-llama/Llama-3.1-8B-Instruct", #there are 1000 models in hugging face so here we are passing which model to use, it is small model of 1B parameter fine tuned on llma model
#     task = "chat-completion", #pass here what we want to do
#     huggingface_api_key=HF_API_KEY    # <--- REQUIRED
# )
# model = ChatHuggingFace(llm = llm)

# result = model.invoke("What is the capital of india")
# print(result.content)


#due to chnages the above  code does not works
from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient # <-- using official working client

load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
print("HF:", HF_API_KEY)  # Debug

model = InferenceClient(
    model = "meta-llama/Llama-3.1-8B-Instruct",
    token=HF_API_KEY  )
#chat_completion requires messages=[...]
result = model.chat_completion(messages=[{"role": "user", "content": "what is capital of india"}]
,max_tokens=50)
# FIX: correct way to access output
print(result.choices[0].message["content"])