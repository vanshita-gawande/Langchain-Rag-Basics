from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(required = True, description="The first number to multiply")
    b: int = Field(required = True, description="The second number to multiply")

def multiply(a: int, b: int) -> int:
    return a * b

#main work , create structured tool , pass function and args schema
multiply_tool = StructuredTool.from_function(
    func = multiply,
    name = "multiply_tool",
    description = "Multiply two Numbers",
    args_schema = MultiplyInput     #most imp pass our class here
)

result = multiply_tool.invoke({"a": 5, "b": 6}) #passing dictionary
print(result)
print(multiply_tool.name)       
print(multiply_tool.description)
print(multiply_tool.args)