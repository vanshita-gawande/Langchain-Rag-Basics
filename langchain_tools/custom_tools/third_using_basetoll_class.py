from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(required = True, description="The first number to multiply")
    b: int = Field(required = True, description="The second number to multiply")

def multiply(a: int, b: int) -> int:
    return a * b

class BaseTollClass(BaseTool):
    name: str = "multiply_tool"
    description: str = "Multiply two Numbers"
    args_schema: Type[BaseModel] = MultiplyInput

#main method name must be same _run
    def _run(self, a: int, b: int) -> int:
        return multiply(a, b)

result = BaseTollClass().invoke({"a": 7, "b": 8}) #passing dictionary
    
print("Custom tool using BaseTool class created successfully.") 
print(result)
print(BaseTollClass().name)       
print(BaseTollClass().description)      
print(BaseTollClass().args)

#here we can make async function of _run also if needed