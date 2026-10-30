from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv 
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

response = model.invoke("what is the capital of india?")
print(response.content)