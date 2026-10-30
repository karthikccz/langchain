from langchain_groq import ChatGroq 
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()
from langchain_core.output_parsers import PydanticOutputParser

model = ChatGroq(model= "openai/gpt-oss-120b",temperature=0.5)
class Person(BaseModel):
    name : str = Field(description =  "name of the person")
    age : int = Field(description = "age of the person")
    facts : list[str] = Field(description = "write 3 facts about the person in a list")
parser = PydanticOutputParser(pydantic_object= Person)

template = PromptTemplate(
    template = " give me name, age and 3 facts about the {topic} \n {format_insturction}",
    input_variables = ['topic'],
    partial_variables = {'format_insturction': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'topic': 'lewis hamilton'})
print(result)