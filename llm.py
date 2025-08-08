from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser,CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
load_dotenv()
API_KEY=os.getenv("OPEN_AI_API_KEY")
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    model="mistralai/mistral-7b-instruct:free",
)


#response = llm.invoke("What is the capital of France?") 
#response = llm.batch("What is the capital of France?")  for multiple inputs
#response = llm.stream("What is the capital of France?") get response in chunks


prompt=ChatPromptTemplate.from_messages(
    [
        ("system", "Extract information from the following phrase. Format the output as a {format}."),
        ("user", "{phrase}?"),
    ]
)
class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    city: str = Field(description="The city where the person lives")

parser=JsonOutputParser(pydantic_object=Person)
# chain =prompt | llm 
chain =prompt | llm | parser

response = chain.invoke({"phrase": "The person is John, 30 years old, living in New York City." , "format": parser.get_format_instructions()})

    
print(response)