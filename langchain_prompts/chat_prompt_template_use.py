from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage

# when want to use dynamic in list of messages or can use ChatPromptTemplate.messages
chat_template = ChatPromptTemplate([
    ('system','you are helpful {domain} expert'),
    ('human','Explain in simple terms, what is {topic}') #dynamic user inputs
])

promt = chat_template.invoke({'domain':'cricket','topic':'batting'})#passing here values for inputs

print(promt)