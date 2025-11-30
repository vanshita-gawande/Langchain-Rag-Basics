from typing import TypedDict

class Person(TypedDict):

    name : str
    age : int

new_person = Person = {'name':'vanshita','age':'9'} #if chnage to string age then also run do not give error

print(new_person)