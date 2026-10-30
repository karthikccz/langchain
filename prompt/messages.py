from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv 
load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen3-4B-Thinking-2507",
                          task="text-generation")
model = ChatHuggingFace(llm=llm)
chat_history = [SystemMessage(content = "You are a helpful assistant") 
] #holds the chat history 

while True:
    user_input = input("user:")
    chat_history.append(HumanMessage(content = user_input)) #append the user input to the chat history
    if user_input.lower() == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content)) #append the model response to the chat history
    print("bot:", result.content)

print("Chat History:", chat_history) #print the entire chat history after the conversation ends

