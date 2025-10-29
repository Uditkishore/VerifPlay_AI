from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from llama_cpp import Llama
import os
import sys
from colorama import Fore, Style, init
import warnings
import logging
from functools import lru_cache
import threading

logging.getLogger("llama_cpp").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=DeprecationWarning)
init(autoreset=True)

# Load vector DB
print(Fore.YELLOW + "Loading vector database...")
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chatbot/embeddings", embedding_function=embedding)
retriever = db.as_retriever(search_kwargs={"k": 3})
print(Fore.GREEN + "Vector DB loaded.")

# Load LLM
print(Fore.YELLOW + "Loading Mistral 7B model...")
model_path = os.path.join("chatbot/models", "phi-2.Q8_0.gguf")
llm = Llama(
    model_path=model_path,
    n_ctx=1024,
    n_threads=6,
    use_mlock=True,
    verbose=False
)
print(Fore.GREEN + "Model loaded. You can now chat with your documents.")

# Caching
@lru_cache(maxsize=128)
def cached_vector_search(query):
    # lru_cache requires hashable arguments, so ensure query is str
    docs = retriever.get_relevant_documents(query)
    # Can't cache doc objects directly, so cache their content
    return tuple(doc.page_content for doc in docs)

llm_cache = {}

def stream_llm_response(prompt, max_tokens=512, stop=None):
    # Generator for streaming LLM responses
    for chunk in llm(prompt, max_tokens=max_tokens, stop=stop, stream=True):
        yield chunk["choices"][0]["text"]

try:
    while True:
        query = input(Fore.CYAN + "\nYou: ")
        if query.strip().lower() in ['exit', 'quit']:
            print(Fore.YELLOW + "Exiting... Goodbye!")
            break

        # Show "thinking..." message
        print(Fore.MAGENTA + "thinking...", end="\r")

        # Caching vector search
        docs_content = cached_vector_search(query)
        docs = [{"page_content": content} for content in docs_content if content.strip()]

        if not docs:
            prompt = f"""
You are a friendly assistant. The user asked something casual or not covered by the documents.
Please respond kindly and helpfully, even without context.

[Question]
{query}

Answer:"""
        else:
            context = "\n".join([doc["page_content"] for doc in docs])
            prompt = f"""
You are a helpful assistant. Use the provided context to answer the user's question.

[Context]
{context}

[Question]
{query}

Answer:"""

        # Caching LLM responses
        cache_key = (prompt,)
        if cache_key in llm_cache:
            answer = llm_cache[cache_key]
            print(Fore.GREEN + "\nBot: " + Style.RESET_ALL + answer)
            continue

        # Streaming partial responses
        print(Fore.GREEN + "\nBot: " + Style.RESET_ALL, end="", flush=True)
        answer_parts = []
        for chunk in stream_llm_response(prompt, max_tokens=512, stop=["\n[Question]", "\nAnswer:", "</s>"]):
            print(chunk, end="", flush=True)
            answer_parts.append(chunk)
        answer = "".join(answer_parts).strip()
        llm_cache[cache_key] = answer
        print()  # Newline after streaming

except KeyboardInterrupt:
    print(Fore.YELLOW + "\n\nExited via keyboard. Bye!")
    sys.exit(0)
