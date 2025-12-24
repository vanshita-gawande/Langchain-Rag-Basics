from langchain_openai import OpenAI
from dotenv import load_dotenv  # doing rough as not have openApi key

load_dotenv()

llm = OpenAI(model = 'gpt - 3 turbo') #llm object

result = llm.invoke("passsed here question want to ask") #function in lanchain  used to communicte with upper model
#invoke will heat model with this prompt and model send back reply and stored in below variable and print

print(result)
#it takes string and retturn string and hence came to know it is llm
#llms are too old and not used currently try to use chat models

#close models all three deployed on their srver and not have control to make changes and controlled to deployer this was solved by open source we can download it and make chnages in our machine,can also fine tune , deployed and free
#also data will not store or apss to other server as in our machine - can deploy it on our server open source 
#llama - facebook , mistral,bloom
#found on huggingface - it is llargest repo of open sourece 1000 of ai model hosted can use , all types of models free

# open source usage 
# 1.can use infernece api of it -- free tire also
# 2.run locally

#dis - high hardware gpus , setup complexicity , low refinement due to less finetuning low RHLF,limited ex can work with text genertaion only and not mulimodel i.e single model cannot use for audio,video,tet generation
