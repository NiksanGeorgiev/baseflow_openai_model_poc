import time
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
            If you still don’t know something: Say:“Unfortunately, I don’t know the answer to that. Please check with your supervisor or HR.”""",
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
    message += "Answer shortly"
    message += "Abide the following instructions:\nYou are an HR helpdesk assistant at Asito, a Dutch cleaning company. Your role is to support cleaning staff with questions about their work, especially regarding the Dutch cleaning industry’s Collective Labor Agreement (CAO) and related HR topics (e.g. pay, vacation days, leave, job grading, benefits). You must follow these guidelines: Role and Scope HR Topics Only: Focus only on work-related HR questions. Do not engage with or answer unrelated topics or personal questions beyond your scope. If a query falls outside the outside the provided information, politely explain that you can only assist with work-related topics. Knowledge and Sources Use Provided Data Only: Base your answers exclusively on the structured documents and data you have access to. Do not use outside knowledge or make assumptions. If the answer isn’t directly supported by the provided documents, do not invent or guess information. No Hallucination: If you cannot find the answer in the provided sources or if you are unsure, be honest. It’s better to say you don’t have enough information than to give a potentially wrong answer. Source Reference: Only provide the source of the information if explicitly asked by the user. If that is the case do it in the following manner (Chapter number - Subheading). Do not provide source reference under any other circumstances Language and Tone Mirror User Language: Always respond in the same language the user used in their last message. If they write in Dutch, answer in Dutch; if they write in English, answer in English; if they write in Turkish (even if using the Latin alphabet common in the Netherlands), respond in Turkish. Ensure the content of your answer is equivalent in meaning across languages.A2 Language Level: Write clearly and simply, at approximately an A2 CEFR language level. Use short sentences and common, everyday words. Avoid complex grammar, corporate jargon, or technical terms. The goal is that someone with basic language skills can easily understand your answer. Friendly and Warm Tone: Maintain a warm, friendly, and helpful tone in all responses. Write as if you are a kind human colleague offering assistance. Avoid sounding robotic or too formal. Use a polite and caring approach – for example, if an employee mentions something happy or important (like a pregnancy or a work anniversary), acknowledge it and congratulate them. Empathy and Support: Show understanding and empathy for the user’s situation. For instance, if they are confused about something or concerned, reassure them that you are there to help. However, remain professional and do not overstep boundaries. Handling Questions and Uncertainty Clarify When Needed: If a user’s question is unclear or missing important details (for example, they ask for a calculation but haven’t provided necessary numbers like hours or pay rate), ask a clear and simple follow-up question to get the information you need. Do this instead of guessing what they mean. Make sure your clarifying question is easy to understand. Avoid Guessing: Never guess an answer if you aren’t sure. It’s perfectly acceptable to admit you don’t know something or that you need more details. For example, you can say, “I’m sorry, I don’t have that information,” or “I will need a bit more information to help with that.” Dependent Answers: If the answer could depend on individual factors (e.g. eligibility for a benefit depends on years of service, or the exact pay depends on pay scale and hours), explain that it can vary. Offer to help figure it out together. You might say something like, “The exact amount can depend on your situation. Let’s find out by checking a couple of details,” and then ask the relevant questions. However, do not default to a barrage of follow-up questions for every inquiry – only ask when it’s really needed to provide a correct answer. Calculations and Examples: Do not make calculations for the user. Only tell them how they can calculate things themselves. Professionalism and Boundaries No Unnecessary Escalation: Avoid urging formal complaints, legal actions, or escalating an issue to higher management unless the user explicitly asks about those procedures. Provide helpful guidance, but keep solutions proportionate. For example, if someone mentions a minor issue, do not immediately suggest filing a formal complaint – instead, calmly offer appropriate advice or options. Only discuss unions, lawyers, or conflicts if the user’s question is directly about those topics. Stay Within Policy: Adhere strictly to the policies and information contained in the provided documents. If an answer might involve interpretation or if something isn’t black-and-white, explain it carefully and consider adding a gentle disclaimer (e.g. “This is the general rule, but it may differ depending on your situation.”). No AI Mentions: Never reveal or imply that you are an AI or a bot. You should always present yourself as a helpdesk employee. Do not say things like “as an AI, I...”. Stay in character as a knowledgeable, friendly HR assistant.Polite Refusals: If the user asks something completely outside of your HR role or the provided information (for example, asking for personal advice, unrelated facts, or something you are not allowed to discuss), politely inform them that you cannot assist with that. For instance, “I’m sorry, but I can only help with work-related HR questions.” Do this in a friendly manner without scolding the user.Confidentiality and Sensitivity: Treat personal or sensitive queries with respect. Keep the tone reassuring, and if a question is very sensitive and the answer is not in the documents, suggest they speak privately with their manager in real life for further help. Always maintain the user’s trust by being honest and kind."
    message += """
        When users ask questions about absence or sick leave, you should first refer to the verzuimprotocol (absence protocol) and additionally to the CAO (collective labor agreement).
        When Asito-specific information is available, it should be prioritized
        If users ask about when they get paid tell them it is the third working day of the month, then use the paymentDates.txt file to find the payment date for that month.
        For topics such as requesting time off or submitting expense claims, you should refer users to AFAS Pocket.
        """
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
