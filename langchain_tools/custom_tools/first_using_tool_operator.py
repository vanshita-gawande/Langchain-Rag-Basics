from langchain_core.tools import tool

#building custom tool

#1.create a function

def multiply(a,b):
    ''' Multiply two numbers'''
    return a * b

#2. add type hints

def multiply(a:int,b:int) -> int:
    '''Multiply two Numbers'''
    return a * b

#3.add decorator to it
#now it becomes magic tool where llm can talk to it
@tool
def multiply(a:int , b:int) -> int:
    '''Multiply two Numbers'''
    return a * b

result = multiply.invoke({"a" : 3, "b" : 4})#passing dictionary

print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)
#llm sees this when we connect it
print(multiply.args_schema.model_json_schema()) #its op is what will be seen by llm