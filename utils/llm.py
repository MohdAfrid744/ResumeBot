import os
import time
from dotenv import load_dotenv
from groq import Groq
from utils.prompt import SYSTEM_PROMPT
from utils.logger import logger
from utils.cache import get_cached_response, set_cached_response

def _get_client_and_model():
    """
    Force reload the .env file and return a fresh Groq client and model name.
    """
    load_dotenv(override=True)
    groq_key = os.environ.get("GROQ_API_KEY")
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    client = Groq(api_key=groq_key)
    return client, model


def get_active_provider_name() -> str:
    """
    Returns a user-friendly string representing the active Groq model.
    """
    _, model = _get_client_and_model()
    return f"Groq ({model})"


def get_response(conversation_history: list[dict]) -> str:
    """
    Send conversation history to Groq and return the assistant's reply.
    Checks cache first.
    """
    cached = get_cached_response(conversation_history)
    if cached is not None:
        return cached

    client, model = _get_client_and_model()
    logger.info(f"Groq LLM request | model={model} | context={len(conversation_history)} msgs")

    # Format conversation history for Groq (System message first, then user/assistant turns)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.6,
            max_tokens=512,
        )
        reply = response.choices[0].message.content.strip()
        elapsed = time.time() - start
        logger.info(f"Groq LLM done | {elapsed:.2f}s | {len(reply)} chars")
        if reply:
            set_cached_response(conversation_history, reply)
        return reply
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Groq LLM error after {elapsed:.2f}s: {e}")
        return f"[Error reaching Groq: {e}]"


def get_response_stream(conversation_history: list[dict]):
    """
    Streaming generator — yields text chunks for use with st.write_stream().
    Checks cache first.
    """
    cached = get_cached_response(conversation_history)
    if cached is not None:
        logger.info("Cache HIT — returning instantly via stream.")
        yield cached
        return

    client, model = _get_client_and_model()
    logger.info(f"Streaming Groq LLM request | model={model} | context={len(conversation_history)} msgs")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    start = time.time()
    full_reply = ""
    try:
        response_stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.6,
            max_tokens=512,
            stream=True
        )
        for chunk in response_stream:
            content = chunk.choices[0].delta.content
            if content:
                full_reply += content
                yield content

        elapsed = time.time() - start
        logger.info(f"Streaming Groq done | {elapsed:.2f}s | {len(full_reply)} chars")
        if full_reply:
            set_cached_response(conversation_history, full_reply)

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Streaming Groq error after {elapsed:.2f}s: {e}")
        yield f"[Error reaching Groq: {e}]"


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """
    Transcribe audio bytes to text using Groq's Whisper endpoint.
    """
    logger.info(f"Transcribing {len(audio_bytes):,} bytes of audio ({filename}) using Groq Whisper…")
    
    client, _ = _get_client_and_model()
    model = os.environ.get("GROQ_AUDIO_MODEL", "whisper-large-v3-turbo")

    try:
        start = time.time()
        transcription = client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model=model
        )
        text = transcription.text.strip()
        elapsed = time.time() - start
        logger.info(f"Groq Whisper transcription succeeded in {elapsed:.2f}s: '{text}'")
        return text
    except Exception as e:
        logger.error(f"Groq Whisper transcription error: {e}")
        return ""
