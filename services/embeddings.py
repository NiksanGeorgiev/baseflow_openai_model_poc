import openai
from typing import List

def get_embeddings_for_chunks(
    chunks: List[str],
    embedding_model: str = "text-embedding-3-small",
    batch_size: int = 50,
) -> List:
    """
    Compute embeddings for a list of text chunks in batches.
    Helps avoid rate limits when calling the OpenAI API.
    """
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"Embedding batch {i} to {i + len(batch) - 1}...")
        response = openai.embeddings.create(
            input=batch, model=embedding_model, timeout=30
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    return embeddings
