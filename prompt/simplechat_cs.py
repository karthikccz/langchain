#here we create a simple dynamic chatbot using the list of messages 

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv 
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
while True:
    user_input = input("user:")
    if user_input.lower() == "exit":
        break
    result = model.invoke(user_input)
    print("bot:", result.content)