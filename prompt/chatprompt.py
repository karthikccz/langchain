from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model= "openai/gpt-oss-120b",temperature=0.5)

prompt = ChatPromptTemplate([
    ("system", "You are a helpful {domain} expert"),
    ("human", "explain in simple terms, what is {topic}")
])

prompt = prompt.invoke({"domain":"AI", "topic":"LLM"})
print(prompt)

response = model.invoke(prompt)
print(response.content)
