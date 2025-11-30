from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('gen_ai.pdf')
data = loader.load()

print(data)
print(len(data))
print(data[0].page_content)
print(data[0].metadata)

#not used with all such as scnned images then need to use other pfd loaders (refer notes)