from typing import List
import tiktoken

from config import EMBEDDING_MODEL, GPT_MODEL_MINI


def num_tokens(text: str, model: str = GPT_MODEL_MINI) -> int:
    """Return the number of tokens in a string using tiktoken."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def split_text_into_chunks(
    text: str, max_tokens: int = 1600, model: str = EMBEDDING_MODEL
) -> List[str]:
    """
    Split text into chunks with at most max_tokens per chunk.
    Splits on double newlines (paragraph breaks) and groups paragraphs until adding
    another would exceed the token limit.
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
