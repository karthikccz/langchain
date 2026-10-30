from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model= "openai/gpt-oss-120b",temperature=0,max_tokens=100)
response = model.invoke("What is the capital of india?")
print(response.content)

