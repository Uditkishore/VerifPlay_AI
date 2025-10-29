from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from load_and_split_docs import load_documents

docs = load_documents()
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(docs, embedding=embedding, persist_directory="chatbot/embeddings")
vectordb.persist()
print("✅ Embedding created and saved.")
