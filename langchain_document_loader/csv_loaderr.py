from langchain_community.document_loaders import CSVLoader

loader = CSVLoader('./datafolder/students_data.csv')

data = loader.load()

print(data)
print(len(data))
print(data[1])
