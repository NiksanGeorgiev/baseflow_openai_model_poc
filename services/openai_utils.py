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
            Language level: All answers must be written at A2 language level. Use simple and clear language. Avoid complicated words. Explain things as if you are talking to someone who is not an office worker. Answer everything very shortly.
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


def create_whatsapp_interactive_message(
    index, embedding_text, df_embeddings, from_number, message_id, answer
):
    """
    Constructs a WhatsApp interactive list message from the possible related questions.

    Returns:
        dict: A dictionary representing the WhatsApp interactive message payload.
    """
    message_body = answer[1:]
    distances, indices = search_index(index, embedding_text, 3)

    print(distances)
    print(indices)
    relevant_questions = [
        [
            question_id
            for distance, question_id in zip(distance_list, index_list)
            if distance < DISTANCE_THRESHOLD
        ]
        for distance_list, index_list in zip(distances, indices)
    ]
    print(relevant_questions)
    if len(relevant_questions[0]) < 1:
        return {
            "messaging_product": "whatsapp",
            "to": from_number,
            "context": {"message_id": message_id},
            "text": {"body": f"{message_body}"},
        }

    results = df_embeddings.iloc[relevant_questions[0]].to_dict(orient="records")
    questions = {result["content"] for result in results}

    # Build rows from options
    rows = [
        {"id": str(i), "title": str(i + 1), "description": option}
        for i, option in enumerate(questions)
    ]

    message_body += "\n\nHowever here are similar questions I can answer:"
    # Construct the final message
    message = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": from_number,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {"text": message_body},
            "action": {
                "sections": [{"title": "Options:", "rows": rows}],
                "button": "Choose option",
            },
        },
    }

    return message
