# ai_router.py

import streamlit as st
import ollama
from google import genai
from google.genai import types

from config import DEFAULT_TEMPERATURE
from groq import Groq
from api_keys import get_groq_api_key, get_gemini_api_key
from safety import register_usage


def debug_provider(provider, model, messages):

    print("=" * 50)
    print(f"Provider : {provider}")
    print(f"Model    : {model}")
#    print(f"Messages : {len(messages)}")
    print("=" * 50)

def ask_ai(messages):

    provider = st.session_state.get("ai_provider", "Lokaal")

    model = st.session_state.get("model_name")

    st.session_state["debug_active_model"] = f"{provider} - {model}"

    if provider == "Gemini":
        return ask_gemini(messages)

    if provider == "Groq":
        return ask_groq(messages)

    return ask_ollama(messages)

def ask_ollama(messages):

    model = st.session_state.get("model_name")

    debug_provider(
        st.session_state.get("ai_provider"),
        model,
        messages
    )

    reply = ollama.chat(
        model=model,
        messages=messages,
        options={
            "temperature": st.session_state.get(
                "temperature",
                DEFAULT_TEMPERATURE
            )
        }
    )

    register_usage(0)

    return reply["message"]["content"].strip()

def ask_gemini(messages):

    api_key = get_gemini_api_key()

    if not api_key:
        return "[Geen Gemini API key gevonden]"

    model = st.session_state.get("model_name")

    client = genai.Client(api_key=api_key)

    debug_provider(
        st.session_state.get("ai_provider"),
        model,
        messages
    )

    prompt = ""

    for msg in messages:

        role = msg["role"]

        if role == "system":
            prompt += f"SYSTEEM:\n{msg['content']}\n\n"

        elif role == "user":
            prompt += f"GEBRUIKER:\n{msg['content']}\n\n"

        else:
            prompt += f"ASSISTENT:\n{msg['content']}\n\n"

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=st.session_state.get(
                "temperature",
                DEFAULT_TEMPERATURE
            )
        )
    )

    try:
        total_tokens = response.usage_metadata.total_token_count
    except Exception:
        total_tokens = 0

    register_usage(total_tokens)

    return response.text.strip()

def ask_groq(messages):

    api_key = get_groq_api_key()

    if not api_key:
        return "[Geen Groq API key gevonden.]"

    model = st.session_state.get("model_name")

    client = Groq(api_key=api_key)

    debug_provider(
        st.session_state.get("ai_provider"),
        model,
        messages
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=st.session_state.get(
            "temperature",
            DEFAULT_TEMPERATURE
        )
    )

    try:
        total_tokens = response.usage.total_tokens
    except Exception:
        total_tokens = 0

    register_usage(total_tokens)

    return response.choices[0].message.content.strip()
