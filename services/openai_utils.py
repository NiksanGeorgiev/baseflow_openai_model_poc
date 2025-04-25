import time
from typing import List, Tuple
import openai
from openai.types.chat import ChatCompletionMessageParam
import pandas as pd
from scipy import spatial
from config import EMBEDDING_MODEL, GPT_MODEL_MINI, TRANSCRIBE_MODEL, DISTANCE_THRESHOLD
from .text_processing import num_tokens
from faiss_index import search_index


def get_embeddings_for_chunks(
    chunks: List[str],
    embedding_model: str = EMBEDDING_MODEL,
    batch_size: int = 50,
) -> List:
    """
    Compute embeddings for a list of text chunks using the OpenAI API.
    Processes the chunks in batches to help avoid rate limits.
    """
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"Embedding batch {i} to {i + len(batch) - 1}...")
        response = openai.embeddings.create(
            input=batch,
            model=embedding_model,
            timeout=30,  # adjust as needed
        )
        # Access the embeddings via attribute access.
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    return embeddings


def strings_ranked_by_relatedness(
    query: str, df: pd.DataFrame, top_n: int = 100
) -> tuple[List[str], List[float]]:
    """
    Given a query and a dataframe with columns "text" and "embedding", compute the query embedding,
    calculate cosine similarity (as relatedness) between the query and each text's embedding,
    and return the top_n texts (and their scores) ranked by relatedness.
    """
    query_response = openai.embeddings.create(
        model=EMBEDDING_MODEL, input=[query], timeout=30
    )
    query_embedding = query_response.data[0].embedding

    # Define relatedness as 1 - cosine distance.
    def relatedness(x, y):
        return 1 - spatial.distance.cosine(x, y)

    strings_and_relatednesses: List[Tuple[str, float]] = [
        (str(row["text"]), relatedness(query_embedding, row["embedding"]))
        for _, row in df.iterrows()
    ]
    sorted_strings = sorted(
        strings_and_relatednesses, key=lambda item: item[1], reverse=True
    )
    top_strings, top_scores = zip(*sorted_strings[:top_n])
    return list(top_strings), list(top_scores)


def query_message(query: str, df: pd.DataFrame, model: str, token_budget: int) -> str:
    """
    Build a message for GPT by retrieving the most relevant text chunks from df.
    It starts with an introduction, then appends articles until the token budget is reached,
    and finally appends the question.
    """
    retrieved_texts, _ = strings_ranked_by_relatedness(query, df, top_n=100)
    introduction = """Use only the articles below to answer the question.
              If the answer cannot be found directly in the articles:
              Do not guess or invent an answer."""
    question = f"\n\nQuestion: {query}. Explain very shortly."
    message = introduction
    for text_section in retrieved_texts:
        next_article = f'\n\nArticle:\n"""\n{text_section}\n"""'
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    df: pd.DataFrame,
    model: str = GPT_MODEL_MINI,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str | None:
    """
    Answer a query by:
    1. Retrieving relevant text sections from df.
    2. Constructing a prompt that includes these sections and the question.
    3. Calling the ChatCompletion endpoint with a system instruction to answer based solely on the provided text.
    """
    message_text = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print("Prompt for GPT:\n", message_text)
    messages: List[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": """
            You are a helpful helpdesk assistant for a cleaning company.
            Purpose: You support cleaning staff with questions about their work, such as vacation days, time off, payslips, working hours, and other HR-related topics.
            Source of information: You only use information from the documents that have been provided to you. If you are not sure about the answer, be honest and say so.
            Language level: All answers must be written at A2 language level. Use simple and clear language. Avoid complicated words.
            Explain things as if you are talking to someone who is not an office worker. Answer everything very shortly.
            Tone: Be friendly, calm, and helpful. Use short sentences and bullet points where it helps with clarity.
            Do not say: Do not invent information. Do not mention that you are an AI.
            Do say: If someone asks where the information came from, refer to the document or say: “According to the document I have received…”
            If you still don’t know something: Say:“😔 Unfortunately, I don’t know the answer to that. Please check with your supervisor or HR.”""",
        },
        {"role": "user", "content": message_text},
    ]
    response = openai.chat.completions.create(
        model=model, messages=messages, temperature=0.17, timeout=30
    )
    return response.choices[0].message.content


def transcribe_audio(audio_file_path: str, model: str = TRANSCRIBE_MODEL) -> str:
    """
    Works with mp3, mp4, mpeg, mpga, m4a, wav, and webm formats.
    The audio file must be less than 25 MB.
    """

    with open(audio_file_path, "rb") as audio_file:
        transcription = openai.audio.transcriptions.create(
            model=model, file=audio_file, response_format="text"
        )

    return transcription


def create_vector_store(name: str) -> str:
    vector_store = openai.vector_stores.create(name=name)
    return vector_store.id


def get_file_id(file_path: str) -> str:
    with open(file_path, "rb") as file_content:
        result = openai.files.create(file=file_content, purpose="assistants")
    return result.id


def add_file_to_vector_store(vector_store_id: str, file_id: str) -> str:
    vector_store_file = openai.vector_stores.files.create(
        vector_store_id=vector_store_id, file_id=file_id
    )
    return vector_store_file.id


def query_vector_store(query: str, vector_store_ids: list[str]) -> str:
    response = openai.responses.create(
        model=GPT_MODEL_MINI,
        input=query,
        tools=[{"type": "file_search", "vector_store_ids": vector_store_ids}],
    )
    return response.output_text


def create_thread():
    thread = openai.beta.threads.create()
    print("Thread created")
    return thread.id


def add_message_to_thread(thread_id, message):
    message += "Answer shortly please."
    message = openai.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=message
    )
    return message.id


def query_assistant(thread_id, assistant_id):
    run = openai.beta.threads.runs.create_and_poll(
        assistant_id=assistant_id,
        thread_id=thread_id,
    )
    while run.status != "completed":
        time.sleep(1)

    messages = openai.beta.threads.messages.list(thread_id=thread_id, order="desc")
    return messages.data[0].content[0].text.value
