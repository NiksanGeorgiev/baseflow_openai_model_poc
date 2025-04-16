import faiss
import numpy as np


def build_faiss_index(embeddings: list):
    """
    Builds a FAISS index (using L2 distance) from a list of embedding vectors.
    """
    embeddings_np = np.array(embeddings).astype("float32")
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings_np)

    print(f"Number of vectors in the index: {index.ntotal}")
    return index


def search_index(index, query_embedding: list, k=5):
    """
    Searches the FAISS index for the k nearest neighbors of the query embedding.
    Returns distances and indices.
    """
    query_np = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query_np, k)
    return distances, indices


def save_faiss_index(index, filename: str):
    """
    Saves the FAISS index to a file.
    """
    faiss.write_index(index, filename)
    print(f"FAISS index saved to {filename}")


def load_faiss_index(filename: str):
    """
    Loads a FAISS index from a file.
    """
    index = faiss.read_index(filename)
    print(f"FAISS index loaded from {filename}")
    return index
