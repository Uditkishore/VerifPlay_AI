from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Dummy summarization function (replace with your own/model)
def summarize_text(text, max_length=3000):
    # For demonstration: just truncate. Replace with real summarization.
    return text[:max_length] + ("..." if len(text) > max_length else "")

def load_documents(doc_folder="chatbot/documents/"):
    all_docs = []
    for file in os.listdir(doc_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(doc_folder, file))
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(os.path.join(doc_folder, file))
        else:
            continue
        docs = loader.load()
        all_docs.extend(docs)

    # Pre-filter: remove documents that are too small or too large
    filtered_docs = [doc for doc in all_docs if 300 < len(doc.page_content) < 8000]

    summarized_docs = []
    for doc in filtered_docs:
        if len(doc.page_content) > 4000:
            doc.page_content = summarize_text(doc.page_content)
        summarized_docs.append(doc)

    # Use moderate chunk size and overlap for speed and context
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    return splitter.split_documents(summarized_docs)
