from sentence_transformers import SentenceTransformer

# Load local EmbeddingGemma
model = SentenceTransformer("./embeddinggemma-300m")

def get_embedding(text):
    """Return embedding vector for given text"""
    return model.encode(text)
