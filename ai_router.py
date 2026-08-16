# ai_router.py

import streamlit as st
import ollama
from google import genai
from google.genai import types

from config import DEFAULT_TEMPERATURE
from groq import Groq
from api_keys import get_groq_api_key, get_gemini_api_key
from safety import register_usage

# Geen Client-klasse meer; we gebruiken direct de ollama-module
OLLAMA_CLIENT = None  # placeholder, niet strikt nodig

# Fallback-modellen per provider als er niks geselecteerd is in st.session_state
DEFAULT_MODELS = {
    "Groq": "openai/gpt-oss-20b",
    "Gemini": "gemini-3.5-flash" 
}

def debug_provider(provider, model, messages):
    print("=" * 50)
    print(f"Provider : {provider}")
    print(f"Model    : {model}")
    print("=" * 50)

def get_active_model(provider):
    """Haalt het geselecteerde model op, of valt terug op de standaardwaarde."""
    model = st.session_state.get("model_name")
    if not model:  # Voorkomt dat model 'None' of een lege string is
        model = DEFAULT_MODELS.get(provider, "qwen2.5:3b")
    return model

def ask_ai(messages):
    provider = st.session_state.get("ai_provider", "Lokaal")
    model = get_active_model(provider)

    st.session_state["debug_active_model"] = f"{provider} - {model}"

    if provider == "Gemini":
        return ask_gemini(messages, model)

    if provider == "Groq":
        return ask_groq(messages, model)

    return NONE #ask_ollama(messages, model)

#def ask_ollama(messages, model=None):
#    if not model:
#        model = get_active_model("Lokaal")

#    debug_provider(
#        st.session_state.get("ai_provider", "Lokaal"),
#        model,
#        messages
#    )

#    reply = OLLAMA_CLIENT.chat(
#        model=model,
#        messages=messages,
#        options={
#            "temperature": st.session_state.get(
#                "temperature",
#                DEFAULT_TEMPERATURE
#            ),
#            "num_gpu": 999,
#            "num_batch": 512,
#            "num_thread": 0,
#        }
#    )

#    register_usage(0)
#    return reply["message"]["content"].strip()

def ask_gemini(messages, model=None):
    if not model:
        model = get_active_model("Gemini")

    api_key = get_gemini_api_key()

    if not api_key:
        return "[Geen Gemini API key gevonden]"

    client = genai.Client(api_key=api_key)

    debug_provider(
        st.session_state.get("ai_provider", "Gemini"),
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

def ask_groq(messages, model=None):
    if not model:
        model = get_active_model("Groq")

    api_key = get_groq_api_key()

    if not api_key:
        return "[Geen Groq API key gevonden.]"

    client = Groq(api_key=api_key)

    debug_provider(
        st.session_state.get("ai_provider", "Groq"),
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
