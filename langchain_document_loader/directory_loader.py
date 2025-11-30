from langchain_community.document_loaders import DirectoryLoader,TextLoader,PyPDFLoader

# ✅ Load TXT files
txt_loader = DirectoryLoader(
    path="datafolder",
    glob="*.txt",
    loader_cls=TextLoader
)

# ✅ Load PDF files
pdf_loader = DirectoryLoader(
    path="datafolder",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

#to print how many files loaded 
# ✅ Load both
txt_docs = txt_loader.load()
pdf_docs = pdf_loader.load()

# ✅ Combine all documents
documents = txt_docs + pdf_docs

#print content of each file

#print(documents)
print(documents[30].page_content)

#AS IT TAKES TOO MUCH TIME SO WE USE LAZY LOAD TECHNIQUE FOR IT