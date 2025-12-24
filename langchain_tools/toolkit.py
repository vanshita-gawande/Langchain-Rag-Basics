#first create two custom functions that are relsted using toll operator
from langchain_core.tools import tool

#function 1
@tool
def add(a: int, b: int) -> int:
    '''Add two numbers'''
    return a + b

#function 2
@tool   
def subtract(a: int, b: int) -> int:
    '''Subtract two numbers'''
    return a - b

#tool kit
class MathToolkit:
    def get_tools(self):
        return [add,subtract] #return name of tools we want as our part

toolkit = MathToolkit()
tools = toolkit.get_tools()

for t in tools:
    print(f"Tool Name: {t.name}")
    print(f"Description: {t.description}")
    print(f"Args: {t.args}")
    print("-----")
