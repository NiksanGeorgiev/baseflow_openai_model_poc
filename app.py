import json
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
from services.file_utils import load_markdown_file
from services.text_processing import split_text_into_chunks
from services.openai_utils import get_embeddings_for_chunks, ask, transcribe_audio

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

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

    try:
        payload = WebhookPayload.model_validate(body)
        print(f"Parsed payload: {payload}")

    except ValidationError as e:
        print(f"Validation error: {e.json()}")
        return jsonify({"error": "Invalid webhook payload"}), 400

    entry = payload.entry[0]
    change = entry.changes[0]
    value = change.value
    message = value.messages[0]
    phone_no_id = value.metadata.phone_number_id
    from_number = message.from_
    message_id = message.id

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
    }

    if message.type not in ["text", "audio"]:
        return jsonify({"error": "Unsupported message type"}), 400

    question = ""
    if message.type == "text":
        question: str = message.text.body
        print(f"Received text message: {question}")
    elif message.type == "audio":
        question: str = handle_audio_message(message, headers)
        print(f"Transcribed text: {question}")

    answer = ask(
        question,
        df,
        token_budget=4096 - 500,
        print_message=False,
    )

    test = requests.post(
        f"https://graph.facebook.com/v22.0/{phone_no_id}/messages",
        json={
            "messaging_product": "whatsapp",
            "to": from_number,
            "context": {"message_id": message_id},
            "text": {"body": f"{answer}"},
        },
        headers=headers,
    )

    return test.json()


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
