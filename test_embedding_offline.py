# test_embedding_offline.py

from chatbot.offline_loader import load_sentence_transformer

print("🚀 Trying to load model completely offline...")

try:
    model = load_sentence_transformer()
    print("✅ Model loaded successfully!")

    # Try embedding a sample sentence
    sample_text = "This is an offline embedding test."
    embedding = model.encode(sample_text)

    print("✅ Embedding generated (first 5 dims):", embedding[:5])
except Exception as e:
    print("❌ Error loading model or generating embedding:")
    print(e)
