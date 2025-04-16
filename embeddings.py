import openai
from config import OPENAI_API_KEY, EMBEDDING_MODEL


openai.api_key = OPENAI_API_KEY


def get_embedding_for_text(text: str, model=EMBEDDING_MODEL):
    """
    Calls the OpenAI embeddings API to get the vector representation of the text.
    """
    response = openai.embeddings.create(model=model, input=text.replace("\n", " "))
    return response.data[0].embedding


def compute_embeddings(df):
    """
    Computes embeddings for each row in the DataFrame.
    Adds a new column 'embedding' to the DataFrame.
    """
    embeddings = []
    for text in df["content"]:
        emb = get_embedding_for_text(text)
        embeddings.append(emb)
    df["embedding"] = embeddings
    return df
