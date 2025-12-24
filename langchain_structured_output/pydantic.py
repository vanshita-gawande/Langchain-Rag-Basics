# from pydantic import BaseModel

# class Student(BaseModel):
#     name : str

# new_student = {'name' :'vanshita'} #it give error when we pass wrong data type and hence can validate data this is not present and possible with typed dict

# student = Student(**new_student) #it gives you the hints

# print(student) #now its type is pydantic,can access using .


from pydantic import BaseModel, ValidationError

# ✅ Pydantic Model (Schema)
class Student(BaseModel):
    name: str
    age: int
    is_active: bool

# ✅ Correct Data (This will work)
new_student = {
    "name": "Vanshita",
    "age": 22,
    "is_active": True
}

student = Student(**new_student)

print("✅ Valid Student Object:")
print(student)
print("Name:", student.name)        # dot access ✅
print("Age:", student.age)
print("Active:", student.is_active)

print("\n-----------------------------\n")

# ❌ Wrong Data (This will raise validation error)
wrong_student = {
    "name": 123,          # ❌ should be str
    "age": "twenty",      # ❌ should be int
    "is_active": "yes"   # ❌ should be bool
}

try:
    student2 = Student(**wrong_student)
except ValidationError as e:
    print("❌ Validation Error:")
    print(e)
#for wrong will give error

#for open ai 
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# ✅ load environment variables
load_dotenv()

# ✅ initialize OpenAI model
model = ChatOpenAI(
    model="gpt-4o-mini",   # lightweight + supports structured output
    temperature=0
)

# ✅ Pydantic Schema
class Student(BaseModel):
    name: str = Field(..., description="Name of the student")
    age: int = Field(..., description="Age of the student")
    is_active: bool = Field(..., description="Whether the student is active or not")

# ✅ Enable structured output
structured_model = model.with_structured_output(Student)

# ✅ Invoke model
result = structured_model.invoke(
    "Create a student record with name Vanshita, age 22, and mark her as active."
)

# ✅ This is now a REAL Pydantic object
print("✅ Structured Output (Pydantic Object):")
print(result)

# ✅ Dot access works
print("\n✅ Access fields using dot notation:")
print("Name:", result.name)
print("Age:", result.age)
print("Active:", result.is_active)

# ✅ Type check proof
print("\n✅ Python Types:")
print(type(result.name))      # str
print(type(result.age))       # int
print(type(result.is_active)) # bool
