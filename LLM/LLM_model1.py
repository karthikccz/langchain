from langchain import OpenAI
from dotenv import load_dotenv

load_dotenv()
model = OpenAI(model="free tier not avaliable")
result =model.invoke("What is the capital of delhi?")
print(result)
