import json
import ast
import os
import requests
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from pydub import AudioSegment
import openai
from pydantic import ValidationError
from schemas import WebhookPayload, WebhookMessage
from config import (
    OPENAI_API_KEY,
    WHATSAPP_ACCESS_TOKEN,
    WHATSAPP_APP_TOKEN,
)
from services.embeddings import compute_embeddings, get_embedding_for_text
from services.file_utils import load_markdown_file
from services.text_processing import split_text_into_chunks
from services.openai_utils import (
    create_whatsapp_interactive_message,
    get_embeddings_for_chunks,
    ask,
    transcribe_audio,
)
from faiss_index import (
    load_faiss_index,
    build_faiss_index,
    save_faiss_index,
)

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

EMBEDDINGS_FILE = os.environ.get("EMBEDDINGS_FILE", "records_and_embeddings.csv")
FAISS_INDEX_FILE = os.environ.get("FAISS_INDEX_FILE", "fais.index")

df_questions = pd.read_csv("questions.csv", sep="|")

# if you want to persist embeddings, check if they exist; else compute them.
if os.path.exists(EMBEDDINGS_FILE):
    print("Loading embeddings from file:", EMBEDDINGS_FILE)
    df_embeddings = pd.read_csv(EMBEDDINGS_FILE, sep="|", lineterminator="\n")

    # If the embeddings column is stored as a string, convert it back to list:
    df_embeddings["embedding"] = df_embeddings["embedding"].apply(ast.literal_eval)
else:
    df_embeddings = compute_embeddings(df_questions.copy())

    # for now, save to CSV
    df_embeddings.to_csv(EMBEDDINGS_FILE, index=False, sep="|")
    print("Embeddings computed and saved to:", EMBEDDINGS_FILE)


embeddings_list = df_embeddings["embedding"].tolist()
print("Embeddings list length:", len(embeddings_list))

# build or load FAISS index from persisted file
if os.path.exists(FAISS_INDEX_FILE):
    faiss_idx = load_faiss_index(FAISS_INDEX_FILE)
else:
    faiss_idx = build_faiss_index(embeddings_list)
    save_faiss_index(faiss_idx, FAISS_INDEX_FILE)
print("FAISS index ready. Index size:", faiss_idx.ntotal)

app = Flask(__name__)
CORS(app)

# --- Data Preparation ---
file_path = "asito_cao.md"
print("Loading markdown file...")
full_text = load_markdown_file(file_path)

print("Splitting text into chunks...")
chunks = split_text_into_chunks(full_text)
print(f"Created {len(chunks)} chunks.")

print("Computing embeddings for chunks...")
embeddings = get_embeddings_for_chunks(chunks)
print("Embeddings computed.")

# Build a DataFrame with text chunks and their embeddings.
df = pd.DataFrame({"text": chunks, "embedding": embeddings})


@app.route("/webhook", methods=["GET"])
def handle_webhook():
    mode = request.args.get("hub.mode")
    challenge = request.args.get("hub.challenge", "").strip().strip('"')
    token = request.args.get("hub.verify_token")

    if mode and token:
        if mode == "subscribe" and token == WHATSAPP_APP_TOKEN:
            return challenge, 200
        else:
            return jsonify({"error": "Invalid token"}), 403
    return jsonify({"error": "Missing parameters"}), 400


@app.route("/webhook", methods=["POST"])
def handle_webhook_post():
    body = request.get_json()
    print(body)

    try:
        payload = WebhookPayload.model_validate(body)
    except ValidationError:
        return jsonify({"error": "Invalid webhook payload"}), 400

    entry = payload.entry[0]
    change = entry.changes[0]
    value = change.value

    if value.statuses:
        return jsonify({}), 204

    if not value.messages or len(value.messages) == 0:
        return jsonify({"error": "No messages found in payload"}), 400

    # Use the first message.
    message = value.messages[0]
    phone_no_id = value.metadata.phone_number_id
    from_number = message.from_
    message_id = message.id

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
    }

    if message.type not in ["text", "audio", "interactive"]:
        return jsonify({"error": "Unsupported message type"}), 400

    question = ""
    if message.type == "text":
        if not message.text:
            return jsonify({"error": "No text content in message"}), 400
        question = message.text.body
    elif message.type == "audio":
        # 'handle_audio_message' expects a WebhookMessage instance.
        question = handle_audio_message(message, headers)
    elif message.type == "interactive":
        try:
            question = message.interactive.list_reply.description
        except Exception as e:
            print(e)

    print(f"Received message: {question}")
    # Mark message as read
    response = requests.post(
        f"https://graph.facebook.com/v22.0/{phone_no_id}/messages",
        json={
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
            "typing_indicator": {"type": "text"},
        },
        headers=headers,
    )

    answer = ask(
        question,
        df,
        token_budget=4096 - 500,
        print_message=False,
    )

    # Send an interactive list as a response
    if "😔" in str(answer):
        question_embedding = get_embedding_for_text(question)
        response = requests.post(
            f"https://graph.facebook.com/v22.0/{phone_no_id}/messages",
            json=create_whatsapp_interactive_message(
                faiss_idx,
                question_embedding,
                df_embeddings,
                from_number,
                message_id,
                answer,
            ),
            headers=headers,
        )
    # Send a regular message as a response
    else:
        response = requests.post(
            f"https://graph.facebook.com/v22.0/{phone_no_id}/messages",
            json={
                "messaging_product": "whatsapp",
                "to": from_number,
                "context": {"message_id": message_id},
                "text": {"body": f"{answer}"},
            },
            headers=headers,
        )

    return jsonify(response.json())


def handle_audio_message(message: WebhookMessage, headers: dict) -> str:
    """
    Process an audio message using the typed Pydantic model.
      1. Download the audio file.
      2. Save it as OGG.
      3. Convert from OGG to WAV.
      4. Transcribe the WAV file using OpenAI.
    Returns the transcribed text.
    """
    # Access attributes directly from the model.
    audio_id = message.audio.id
    # Get audio info from WhatsApp Graph API.
    audio_info = requests.get(
        f"https://graph.facebook.com/v22.0/{audio_id}", headers=headers
    )
    media_id = audio_info.url
    media_url_content = requests.get(media_id, headers=headers).content.decode("utf-8")
    media_url_json = json.loads(media_url_content)
    media_url = media_url_json.get("url")
    audio_response = requests.get(media_url, headers=headers).content

    if not audio_response:
        app.logger.error("Failed to download voice note")
        return ""

    ogg_filename = f"{audio_id}.ogg"
    try:
        with open(ogg_filename, "wb") as audio_file:
            audio_file.write(audio_response)
    except Exception as e:
        app.logger.error("Could not save voice note: %s", e)
        return ""

    if os.path.getsize(ogg_filename) == 0:
        app.logger.error("Downloaded file is empty")
        return ""

    wav_filename = f"{audio_id}.wav"
    try:
        audio = AudioSegment.from_file(ogg_filename, format="ogg")
        audio.export(wav_filename, format="wav")
    except Exception as e:
        app.logger.error("Failed converting/exporting voice note: %s", e)
        return ""

    return transcribe_audio(wav_filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
