# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# from transformers import pipeline
# from dotenv import load_dotenv

# load_dotenv()

# # Create HF text2text pipeline
# pipe = pipeline(
#     "text-generation",
#     model="TinyLlama/TinyLlama-1.1B-Chat-v0.3",
#     max_new_tokens=150,
#     temperature=0.7
# )

# # Wrap inside LangChain pipeline
# llm = HuggingFacePipeline(pipeline=pipe)

# # Convert to chat model
# model = ChatHuggingFace(llm=llm)
# #
# chat_history = []
# # Simple chat loop
# while True:
#     user_input = input("You: ")
#     chat_history.append(user_input)
#     if user_input.lower() == "exit":
#         break
#     result = model.invoke(chat_history)
#     chat_history.append(result.content)
#     print("AI:", result.content)
# print(chat_history)


#using token 
# from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# import os

# load_dotenv()

# repo_id = "mistralai/Mistral-7B-Instruct-v0.2"   # example model

# llm = HuggingFaceEndpoint(
#     repo_id=repo_id,
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
#     temperature=0.7,
#     max_new_tokens=150
# )

# model = ChatHuggingFace(llm=llm)

# while True:
#     text = input("You: ")
#     if text == "exit":
#         break
#     res = model.invoke(text)
#     print("AI:", res.content)

#it does not save or remember our chats hence we append it in chat_history = [] like but then it give the result in dict format into single array type which does not differentiate who is user and who is ai ["hi","hello","one","two"] like this and is NOT a proper chat history format. hence we are using the concept of messages 


#using messages

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
import os

load_dotenv()

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"   # example model

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    temperature=0.7,
    max_new_tokens=150
)

model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content="you are helpgul ai asistant") #by defult the first message given to model
]

while True:
    user_input = input("You: ")
    
    if user_input == "exit":
        break
    chat_history.append(HumanMessage(content = user_input))
    # gen ai response
    res = model.invoke(chat_history)
    #add ai message to mememory
    chat_history.append(AIMessage(content=res.content))

    print("AI:", res.content)
    print("\nHistory:", chat_history)