from langchain_groq import ChatGroq 
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

model  = ChatGroq(model= "openai/gpt-oss-120b",temperature=0.5)
schema = [ ResponseSchema(name = 'fact1', description = 'write a fact1 about topic \n'),
          ResponseSchema(name = 'fact2', description = 'write a fact2 about topic \n'),
          ResponseSchema(name = 'fact3', description = 'write a fact3 about topic \n')
]
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = "give me 3 facts about {topic} \n {format_instruction}",
    input_variables = ['topic'],
    partial_variables = {'format_instruction': parser.get_format_instructions()})

chain = template | model | parser
result = chain.invoke({'topic': 'lewis hamilton'})
print(result)