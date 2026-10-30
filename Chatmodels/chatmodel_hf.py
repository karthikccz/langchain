#open model 1
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct",
                          task="text-generation")
model = ChatHuggingFace(llm=llm)
result = model.invoke("who is GOAT in F1")
print(result.content)
# we remotely use the huggingface repo model Qwen/Qwen2.5-7B-Instruct for text generation tasks. The model is accessed through the 
# HuggingFaceEndpoint, which allows us to interact with the model hosted on Hugging Face's platform.