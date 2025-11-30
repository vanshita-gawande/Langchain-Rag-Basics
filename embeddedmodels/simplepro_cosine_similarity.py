#creating a project that retire user query response based on semantic search
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

embedding = HuggingFaceEmbeddings(model = 'sentence-transformers/all-MiniLM-L6-v2')

documents = "Virat Kohli – A modern batting legend known for his consistency, aggressive style, and unmatched chase-mastery across formats.",
"Rohit Sharma – Famous for his effortless timing and multiple ODI double centuries, making him one of the most elegant openers in cricket.",
"MS Dhoni – A calm and strategic leader recognized worldwide for his finishing ability and guiding India to major ICC trophies.",
"Sachin Tendulkar – The “God of Cricket,” celebrated for his extraordinary skill, longevity, and countless batting records.",
" Jasprit Bumrah – A world-class fast bowler known for his unplayable yorkers, unique action, and match-winning spells in all formats."

query = 'Tell me about rohit sharma' #and we want to give ans from the baove documents

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

#so we first find document embedding then user query then apply seaech

scores = cosine_similarity([query_embedding],doc_embeddings)[0] #pass here 2d list and op also 2d so we convert it firtsly into 1 d
#list(enumerate(scores)) #attacehed index no it so help during sorting on basis on similarity scores

index , score = sorted(list(enumerate(scores)),key = lambda x:x[1])[-1] # lambda : It selects the index and value of the maximum score.
print(query)
print("similarity score",score)
print(documents[index]) #on basis of score from documentes, it will print the value based on maximum score and the sring present on this index

# tell me about virat kolhi
#similarity score 0.3137494514631395
#Virat Kohli – A modern batting legend known for his consistency, aggressive style, and unmatched chase-mastery across formats.

#simply here we are doing semantic search and jo document jyda similar hai usko nikal ke la rahe hai
#here everytime we generate the embeddings which is the costly process so we store it in database called vectore database all the ambeddings also called as retieval process
#this is the commom produre and as per the application complexicity increases
#hence embeddings models are useful to perform semantic search