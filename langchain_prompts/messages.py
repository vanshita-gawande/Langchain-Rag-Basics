# Chat models (like ChatGPT, Llama-Chat, TinyLlama-Chat, Mistral-Instruct, Phi-3-mini, etc.)
# do not take a single string.
# They need structured conversation, like:
# who is talking? (user or AI)
# what was said before?
# what is the full context?
# That structure is given using messages.
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",      # cloud model
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    temperature=0.7,
    max_new_tokens=200
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about India.")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print("AI:", result.content)
print("\nHistory:", messages)


#now using this concept in our chatbot