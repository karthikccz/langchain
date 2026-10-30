from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


model = ChatGroq(model= "openai/gpt-oss-120b",temperature=0)
template1 = PromptTemplate(
    template = "write a detailed report on {topic}",
    input_variables = ["topic"]
)
template2 = PromptTemplate(
    template = "write a 5 line report on following text. \n {text}",
    input_variables = ["text"]
)
parser = StrOutputParser()

result = template1 | model | parser | template2 | model | parser
print(result.invoke({'topic':'lewis hamilton'}))


'''prompt1 = template1.invoke({'topic': 'lewis hamilton'})
result = model.invoke(prompt1)
print(result.content)
prompt2 = template2.invoke({'text': result.content})
result1 = model.invoke(prompt2)
print(result1.content)'''