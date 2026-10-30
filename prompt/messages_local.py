from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFacePipeline.from_model_id(model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                        task = "text-generation",
                                        pipeline_kwargs= {"temperature":0.5, "max_new_tokens":100})
model = ChatHuggingFace(llm=llm)
chat_history = [SystemMessage(content = "You are a helpful assistant")]

while True:
    user_input = input("User:")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("Bot:", result.content)
print("chat history:", chat_history)    
