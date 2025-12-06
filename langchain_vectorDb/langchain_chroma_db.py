# A Vector Database:
# ✅ Stores these vectors
# ✅ Finds most similar meaning
# ✅ Used in ChatGPT, RAG, Search, Recommenders

# 2️⃣ WHAT IS CHROMA DB?
# ChromaDB is a free, local vector database used for:
# ✅ RAG projects
# ✅ PDF Question Answering
# ✅ AI Search Engines
# ✅ Chatbots with Memory

# FULL ✅ WORKING BEGINNER-FRIENDLY CHROMA IMPLEMENTATION
# (Using HuggingFace + Your Token)
# This example will:
# ✅ Take text
# ✅ Split into chunks
# ✅ Convert into embeddings
# ✅ Store in ChromaDB
# ✅ Ask a question
# ✅ Get answer from stored data
# ALSO REFER NOTES FOR REVISE

from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")#till now in all prev we using acess token which write but here as fetching the ans we need read hence create new token from hugface portal and name it api token thats why getting error as langchain not finding it as Your library was looking for HUGGINGFACEHUB_API_TOKEN but you only had HUGGINGFACEHUB_ACCESS_TOKEN
if not token:
    raise ValueError("not found env")
else:
    print(f"✅ Token loaded: {token[:10]}...") 

# # ✅ Set the token explicitly
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = token  # Some versions use this key

#create a sample file if not exist and add data to it
with open("rag_data.txt","w",encoding="utf-8") as f:
    f.write('''Artificial Intelligence is the simulation of human intelligence.
Machine Learning is a subset of Artificial Intelligence.
Deep Learning is a subset of Machine Learning that uses neural networks.
Virat Kohli is one of the best cricketers in the world.
India is one of the fastest growing AI markets.''')

print("file created successfully")  

#now load the document with loader
loader = TextLoader('rag_data.txt',encoding='utf-8')
documents = loader.load()

#chuning , split the document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20
)

#get chunks with above splitter
chunks = splitter.split_documents(documents)

#create the embeddings using hf free model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#now store them in vector db
vectore_store = Chroma.from_documents(
    documents= chunks,
    embedding=embeddings,
    persist_directory="chroma_db" #name of the store we want,can pass any dir name
)

print("\nDATA STORED IN CHROMA DB SUCCESSFULLY!\n")

#create retriever
retriever = vectore_store.as_retriever()

#load llm for question answers
base_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.3,
    max_new_tokens=200,
)

llm = ChatHuggingFace(llm = base_llm)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Use the context to answer. "
            "If the answer is not in the context, say you don't know.",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}",
        ),
    ]
)

parser = StrOutputParser()
#create rag chain
rag_chain =({
    "question":RunnablePassthrough(),
    "context":retriever,
}
| rag_prompt | llm | parser
)

#ask questions
question = "Who is Virat Kohli?"
answer = rag_chain.invoke(question)

print("\nQUESTION:", question)
print("\nANSWER FROM CHROMA DB:\n", answer)#give the answer based on our context i.e our rag_file.txt


# If later you want to move beyond Chroma:
# FAISS (local, in-memory, very fast)
# Qdrat (open-source, server or cloud)
# Pinecone (managed SaaS, easy but paid at scale)
# Weaviate (cloud + self-host options)
# pgvector (Postgres extension, great for production DBs) 
# But for now, Chroma is perfect for your LangChain + HF learning.