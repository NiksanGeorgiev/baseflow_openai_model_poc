import json
import os
from typing import List

import openai
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from pydub import AudioSegment
from scipy import spatial
import tiktoken

# Load environment variables from .env file
load_dotenv()

# Access the variables
openai.api_key = os.getenv("OPENAI_API_KEY")
whatsapp_access_token = os.getenv("WHATSAPP_ACCESS_TOKEN")
whatsapp_app_token = os.getenv("WHATSAPP_APP_TOKEN")


# Define the models we'll use:
GPT_MODELS = ["gpt-4o", "gpt-4o-mini"]  # Choose one for answering
EMBEDDING_MODEL = "text-embedding-3-small"
# --- 1. Load the Markdown File --- 
def load_markdown_file(file_path: str) -> str:
    """Load the entire markdown file as a string."""
    with open(file_path, "r", encoding="utf-8") as f: 
        return f.read()

# --- 2. Token Counting & Text Chunking ---
def num_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Return the number of tokens in a string using tiktoken."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def split_text_into_chunks(
    text: str, max_tokens: int = 1600, model: str = "text-embedding-3-small"
) -> List[str]:
    """
    Split the text into chunks such that each chunk has at most max_tokens.
    The function splits on double newlines (assuming paragraph breaks) and groups paragraphs
    until adding one more would exceed the token limit.
    """
    encoding = tiktoken.encoding_for_model(model)
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = current_chunk + "\n\n" + para if current_chunk else para
        if num_tokens(candidate, model) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
            # If the paragraph itself is too long, truncate it.
            if num_tokens(para, model) > max_tokens:
                truncated = encoding.decode(encoding.encode(para)[:max_tokens])
                chunks.append(truncated)
                current_chunk = ""
            else:
                current_chunk = para
        else:
            current_chunk = candidate
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


# --- 3. Compute Embeddings for Each Chunk ---
def get_embeddings_for_chunks(
    chunks: List[str],
    embedding_model: str = "text-embedding-3-small",
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


# --- 4. Build Search Functions ---
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

    strings_and_relatednesses = [
        (row["text"], relatedness(query_embedding, row["embedding"]))
        for _, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    top_strings, top_scores = zip(*strings_and_relatednesses[:top_n])
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
        "If the answer cannot be found in the articles, write 'I could not find an answer.'"
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

def transcribe_audio(audio_file_path: str)-> str:
    """
    Works with mp3, mp4, mpeg, mpga, m4a, wav, and webm formats.
    The audio file must be less than 25 MB.
    """ 

    with open(audio_file_path, "rb") as audio_file:
        transcription = openai.audio.transcriptions.create(
            model="gpt-4o-transcribe", 
            file=audio_file, 
            response_format="text"
        )

    return transcription


def ask(
    query: str,
    df: pd.DataFrame,
    model: str = GPT_MODELS[1],
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """
    Answer a query by:
    1. Retrieving relevant text sections from df.
    2. Constructing a prompt that includes these sections and the question.
    3. Calling the ChatCompletion endpoint with a system instruction to answer based solely on the provided text.
    """
    message_text = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print("Prompt for GPT:\n", message_text)
    messages = [
        {
            "role": "system",
            "content": "You answer questions based on the provided articles.",
        },
        {"role": "user", "content": message_text},
    ]
    response = openai.chat.completions.create(
        model=model, messages=messages, temperature=0, timeout=30
    )
    return response.choices[0].message.content


app = Flask(__name__)
CORS(app)

# 1. Load the scraped Markdown file.
file_path = "asito_cao.md"
print("Loading markdown file...")
full_text = load_markdown_file(file_path)

# 2. Split the text into manageable chunks.
print("Splitting text into chunks...")
chunks = split_text_into_chunks(full_text)
print(f"Created {len(chunks)} chunks.")

# 3. Compute embeddings for each chunk.
print("Computing embeddings for chunks...")
embeddings = get_embeddings_for_chunks(chunks)
print("Embeddings computed.")

# 4. Build a DataFrame to store text chunks and their embeddings.
df = pd.DataFrame({"text": chunks, "embedding": embeddings})


@app.route("/ask", methods=["POST"])
def handle_question():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    user_query = data["question"]
    answer = ask(
        user_query,
        df,
        model=GPT_MODELS[1],
        token_budget=4096 - 500,
        print_message=False,
    )

    return jsonify({"question": user_query, "answer": answer})

@app.route("/webhook", methods=["GET"])
def handle_webhook():
    mode = request.args.get("hub.mode")
    challange = request.args.get("hub.challenge").strip().strip('"')
    token = request.args.get("hub.verify_token")

    if mode and token:
        if mode == "subscribe" and token == whatsapp_app_token:
            return challange, 200
        else:
            return jsonify({"error": "Invalid token"}), 403
        

@app.route("/webhook", methods=["POST"])
def handle_webhook_post():
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {whatsapp_access_token}",
    }
    body = request.get_json()
    print(f"Received body: {body}")
    question = None

    if body.get("object"):
        entry = body.get("entry", [])
        if (
            entry
            and entry[0].get("changes")
            and entry[0]["changes"][0].get("value", {}).get("messages")
            and entry[0]["changes"][0]["value"]["messages"][0]
        ):
            message = entry[0]["changes"][0]["value"]["messages"][0]
            phon_no_id = entry[0]["changes"][0]["value"]["metadata"]["phone_number_id"]
            from_number = message["from"]
            message_id = message["id"]

            if message["type"] == "text":
                # Handle text messages
                question = message["text"]["body"]

            elif message["type"] == "audio":
                # Handle audio messages
                audio_id = message["audio"]["id"]

                # Download the audio file
                audio_info = requests.get(
                        f"https://graph.facebook.com/v22.0/{audio_id}",
                        headers=headers
                    )

                media_id = audio_info.url

                media_url_content = requests.get(media_id, headers=headers).content.decode("utf-8")
                media_url_json = json.loads(media_url_content)

                # Extract the URL
                media_url = media_url_json.get("url")
                audio_response = requests.get(media_url, headers=headers).content

                if audio_response != None:
                    try:
                        with open(f"{audio_id}.ogg", "wb") as audio_file:
                            audio_file.write(audio_response)
                    except Exception as e:
                        return jsonify({"error": "Could not read voice note"}), 400
                    if os.path.getsize(f"{audio_id}.ogg") == 0:
                        return jsonify({"error": "Downloaded file is empty"}), 400
                else:
                    print(f"Failed to download voice note: {audio_response.text}")

                # Convert OGG to WAV
                try:
                    audio = AudioSegment.from_file(f"{audio_id}.ogg", format="ogg")
                    audio.export(f"{audio_id}.wav", format="wav")
                except Exception as e:
                    return jsonify({"error": "Failed converting or exporting voice note"}), 400
                
                question = transcribe_audio(f"{audio_id}.wav")

            print(f"Transcribed text: {question}")
            answer = ask (
                question,
                df,
                model=GPT_MODELS[1],
                token_budget=4096 - 500,
                print_message=False,
            )

            requests.post(
                f"https://graph.facebook.com/v22.0/{phon_no_id}/messages",
                json = {
                    "messaging_product": "whatsapp",
                    "to": from_number,
                     "context": {
                        "message_id": message_id
                    },
                    "text": {
                        "body": f"{answer}"
                    },
                },
                headers = headers
            )

            return jsonify({"status": "success"}), 200
        else:
            return "", 204
    else:
        return jsonify({"error": "Invalid body param"}), 404            

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
