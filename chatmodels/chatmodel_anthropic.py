#claude by anthropic
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model = "cluade")
result = model.invoke("passs here quaetion")

print(result.content)
