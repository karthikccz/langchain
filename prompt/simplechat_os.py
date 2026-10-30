from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv 
load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",
                          task="text-generation")
model = ChatHuggingFace(llm=llm)
chat_history = [] #holds the chat history 

while True:
    user_input = input("user:")
    chat_history.append(user_input) #append the user input to the chat history
    if user_input.lower() == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(result.content) #append the model response to the chat history
    print("bot:", result.content)

print("Chat History:", chat_history) #print the entire chat history after the conversation ends


#here we manually gave the names for user and bot, and sometimes we dont know how answerd and show questioned to over come this we use "messages".
