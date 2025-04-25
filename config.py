import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_APP_TOKEN = os.getenv("WHATSAPP_APP_TOKEN")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")

# Models configuration
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL_MINI = "gpt-4o-mini"
TRANSCRIBE_MODEL = "gpt-4o-transcribe"

# Embedding and chunking parameters
TARGET_TOKENS = 512

DISTANCE_THRESHOLD = 0.9
