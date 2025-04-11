from typing import List, Tuple
import openai
from openai.types.chat import ChatCompletionMessageParam
import pandas as pd
from scipy import spatial
from config import EMBEDDING_MODEL, GPT_MODEL_MINI, TRANSCRIBE_MODEL
from .text_processing import num_tokens


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
            input=batch, model=embedding_model, timeout=30  # adjust as needed
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
    introduction = (
        "Use the below articles to answer the subsequent question. "
        "If the answer cannot be found in the articles, try to formulate an question that you can answer with the articles and that is related to the original question of the user. If you find such questions, provide them in a list and in the form that they need to be asked in. If this is not possible and there isn't anything related just say 'Unfortunately, I don’t know the answer to that. Please check with your supervisor or HR.'"
    )
    question = f"\n\nQuestion: {query}"
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
            "content": "You are a helpful helpdesk employee for a cleaning company. Purpose: You support cleaning staff with questions about their work, such as vacation days, time off, payslips, working hours, and other HR-related topics.Source of information: You only use information from the documents that have been provided to you. If you are not sure about the answer, be honest and say so.Language level: All answers must be written at B1 language level. Use simple and clear language. Avoid complicated words. Explain things as if you are talking to someone who is not an office worker.Tone: Be friendly, calm, and helpful. Use short sentences and bullet points where it helps with clarity.Do not say: Do not invent information. Do not mention that you are an AI.Do say: If someone asks where the information came from, refer to the document or say: “According to the document I have received…”If you don’t fully understand the question: Ask a clear and simple follow-up question to better understand what the user means. Do not guess the answer.If you still don’t know something: Say: “Unfortunately, I don’t know the answer to that. Please check with your supervisor or HR.”",
        },
        {"role": "user", "content": message_text},
    ]
    response = openai.chat.completions.create(
        model=model, messages=messages, temperature=0, timeout=30
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

def create_whatsapp_interactive_message(text, from_number):
    """
    Constructs a WhatsApp interactive list message from a given text input.

    The input text should include a body message followed by a list of options,
    where each option starts with "- ".

    Returns:
        dict: A dictionary representing the WhatsApp interactive message payload.
    """
    # Split the input text into body and options
    lines = text.strip().splitlines()
    body_lines = []
    options = []

    in_options = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            in_options = True
            options.append(stripped[2:].strip())  # Remove the "- " prefix
        elif not in_options:
            body_lines.append(stripped)

    message_body = " ".join(body_lines)

    # Build rows from options
    rows = [
        {"id": str(i), "title": option}
        for i, option in enumerate(options)
    ]

    # Construct the final message
    message = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": from_number,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {
                "text": message_body
            },
            "action": {
                "sections": [
                    {
                        "title": "Choose from the following options",
                        "rows": rows
                    }
                ],
                "button": "Choose option"
            }
        }
    }
    print(message_body)
    print(rows)

    return message
