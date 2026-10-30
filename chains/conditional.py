from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
load_dotenv()

model = ChatGroq(model= "openai/gpt-oss-120b",temperature=0.5)
parser1=StrOutputParser()

class Feedback(BaseModel):
    Sentiment : Literal['positive','negative'] = Field(description = "sentiment of the feedback")

parser = PydanticOutputParser(pydantic_object = Feedback)


prompt1 = PromptTemplate(
    template = "give me a feed back about the {feedback} \n {format_instruction}",
    input_variables = ['feedback'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser

prompt2 = PromptTemplate(
    template = "give me the appropriate response to this positive feedback as a human being in one paragraph:\n{feedback}",
    input_variables = ['feedback']
)
prompt3 = PromptTemplate(
    template = "give me the appropriate response to this negative feedback as a human being in one paragraph:\n{feedback}",
    input_variables = ['feedback']
)

condition_chain = RunnableBranch(
    (lambda x : x.Sentiment == 'positive', prompt2 | model |parser1),
    (lambda x : x.Sentiment == 'negative', prompt3 | model | parser1),
    RunnableLambda(lambda x : "invalid feedback")
)

chain = classifier_chain | condition_chain
result = chain.invoke({'feedback': 'damn the movie was worst!'})
print(result)
chain.get_graph().print_ascii()