import openai
import os
from config import OPENAI_API_KEY, VECTOR_STORE_ID

client = openai.Client(api_key=OPENAI_API_KEY)

for filename in os.listdir("files"):
    file_path = os.path.join("files", filename)
    if os.path.isfile(file_path):
        client.vector_stores.files.upload_and_poll(
            vector_store_id=VECTOR_STORE_ID,
            file=open(file_path, "rb")
        )
