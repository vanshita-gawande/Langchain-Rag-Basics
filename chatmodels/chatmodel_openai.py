from langchain_openai import ChatOpenAI  #basechat model father of all and all are inhetit from it
#llm inherit - base llm and basechta model for chatmodel
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model ="gpt-4",temperature=0.2,my_completion_tokden=10)#value 0 to 2 creativity parameter,for coding keep in 0 side and for poem ceative 1 side
#get the token in output not -- they are equal the words roughly
resultfrommodel = model.invoke("pass here string")
print(resultfrommodel.content) #to get specific answer