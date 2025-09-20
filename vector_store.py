import faiss
import numpy as np
import pickle
from embedder import get_embedding

INDEX_PATH = "embeddings/student.index"
CHUNKS_PATH = "embeddings/chunks.pkl"

def add_chunks(chunks):
    """Add new text chunks to FAISS index"""
    vectors = np.array([get_embedding(c) for c in chunks])
    dimension = vectors.shape[1]

    try:
        index = faiss.read_index(INDEX_PATH)
        existing_chunks = pickle.load(open(CHUNKS_PATH, "rb"))
    except:
        index = faiss.IndexFlatL2(dimension)
        existing_chunks = []

    index.add(vectors)
    faiss.write_index(index, INDEX_PATH)

    all_chunks = existing_chunks + chunks
    pickle.dump(all_chunks, open(CHUNKS_PATH, "wb"))

def search_query(query, top_k=3):
    """Search for top_k relevant chunks"""
    query_vec = np.array([get_embedding(query)])
    index = faiss.read_index(INDEX_PATH)
    chunks = pickle.load(open(CHUNKS_PATH, "rb"))
    D, I = index.search(query_vec, top_k)
    results = [chunks[i] for i in I[0]]
    return results
