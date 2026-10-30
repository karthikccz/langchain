from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = ChatGroq(model= "openai/gpt-oss-120b",temperature=0.5)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template =  "give me description about {text}",
    input_variables=["text"])

prompt2 = PromptTemplate(
    template = "give the top interesting facts about the {text}",
    input_variables=["text"]
)
chain = prompt1 | model |parser | prompt2 | model | parser
result = chain.invoke({'text': 'lewis hamilton'})
print(result)