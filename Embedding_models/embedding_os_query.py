from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text = "the capital of india is delhi"
vector = embedding.embed_query(text)
#for document embedding
#vector = embedding.embed_documents([text])
print(str(vector))