from dotenv import load_dotenv
import os
import uuid

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ✅ Load .env
load_dotenv()

# create embedding 
embeddings = HuggingFaceEmbeddings(
     model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#load or craete chroma db
vectorstore = Chroma(
    persist_directory="my_crud_db",
    embedding_function=embeddings
)

print("chroma db loaded successfully")

def create_document(text: str):
    doc_id = str(uuid.uuid4()) #autogenerate unique id

    vectorstore.add_texts(
        texts=[text],
        ids=[doc_id],
        metadatas=[{"source":"manual_insert"}]
    )
    print("Doocument Created")
    print("ID",doc_id)
    return doc_id

#read (search document)
def search_document(query:str):
    results = vectorstore.similarity_search(query,k=2) #k is for give the relevant documents
    print("search results")
    for i,doc in enumerate(results):
        print("Content is ",doc.page_content)
        print("Metadata",doc.metadata)

# UPDATE (MODIFY EXISTING DOCUMENT BY ID)

def update_document(doc_id: str,new_text: str):
    #firstdelete the old version
    vectorstore.delete(ids = [doc_id])

    #re-insert updated version with same id
    vectorstore.add_texts(
        texts = [new_text],
        ids = [doc_id],
        metadatas=[{"source":"updated"}]
    )
    print("document updated")
    print("id",doc_id)

# ✅ DELETE (REMOVE DOCUMENT BY ID)

def delete_document(doc_id: str):
    vectorstore.delete(ids = [doc_id])
    print("document deleteed")
    print("id",doc_id)


if __name__ == '__main__':
    doc_id = create_document("Virat Kohli is one of the greatest cricketers in the world.")

    search_document("who is virat kolhi?")

    update_document(
        doc_id,
        "Virat Kohli is a former Indian captain and one of the greatest modern-era batsmen."
    )

    #read again after update
    print("\nafter update")
    search_document("Virat kolhi")

    #delete
    delete_document(doc_id)

    #read after delete
    print("\nafter delete")
    search_document("virat kolhi")  #You still have OLD DATA in the same DB folder ,That’s why search still returns results.