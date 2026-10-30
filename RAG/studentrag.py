from langchain_community.document_loaders import PyPDFLoader, TextLoader    
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import StrOutputParser

#data loading
loader = TextLoader(r"C:\Users\karth\Desktop\Langchain\RAG\chapter.txt", encoding="utf-8")
documents = loader.load()


#chunking
splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 150)
chunks = splitter.split_documents(documents)

#embedding
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#vector store
vector_store = FAISS.from_documents(chunks,embedding)
print("vector store is created!!")