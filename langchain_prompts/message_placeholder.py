from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder


# create empty chat history list
chat_history = []
#chat Template
chat_template = ChatPromptTemplate.from_messages([
    ('system','you are helpful customer support agents'),
    #now when pass query the model does not have context of previous chat hence create placeholder
    MessagesPlaceholder(variable_name='c'),
    ('human','query')
]
)
#load chat history
with open('chat_history.txt') as f:
   chat_history.extend(f.readlines())
print(chat_history)

#current user query
ch = 'Where is my order?'

# create prompt
prompt = chat_template.invoke({'c':chat_history,
'query' : ch
})

print(prompt)
#it loads the previous chat first and then give response to our query