# chat.py — Hugging Face compatibele versie
import logging
import json
from pathlib import Path
import streamlit as st
import requests
import google.genai as genai

from config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    GEMINI_API_KEY,
    PROFILE_FILENAME,
)

from documents import retrieve_context
from utils import log_event

# ---------------------------------------------------------
# 1. ComfyUI uitschakelen (werkt niet op Hugging Face)
# ---------------------------------------------------------
def generate_image_comfy(prompt: str):
    st.warning("ComfyUI beeldgeneratie is niet beschikbaar op Hugging Face Spaces.")
    return None

# ---------------------------------------------------------
# 2. Persona-afbeelding laden
# ---------------------------------------------------------
def get_persona_image():
    choice = st.session_state.get("active_persona", "Eva Lumen")

    image_map = {
        "Eva Lumen": "Eva.jpg",
        "Astraea": "Astraea.jpg",
        "Standaard": "default.jpg"
    }

    filename = image_map.get(choice, "default.jpg")
    return Path(__file__).parent / "images" / filename

# ---------------------------------------------------------
# 3. Profiel laden
# ---------------------------------------------------------
def load_profile():
    choice = st.session_state.get("active_persona", "Eva Lumen")

    profile_map = {
        "Eva Lumen": "eva_profile.json",
        "Astraea": "astraea_profile.json",
        "Standaard": "default_profile.json"
    }

    filename = profile_map.get(choice, "default_profile.json")
    profile_path = Path(__file__).parent / "profiles" / filename

    if not profile_path.exists():
        logging.warning(f"Profielbestand niet gevonden: {profile_path}")
        return "", ""

    try:
        with profile_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("persona", ""), data.get("description", "")
    except Exception as e:
        logging.exception(f"Kon profiel niet laden: {e}")
        return "", ""

# ---------------------------------------------------------
# 4. Gemini API key ophalen
# ---------------------------------------------------------
def get_gemini_api_key():
    user_provided_key = st.session_state.get("gemini_api_key_user", "")
    if user_provided_key:
        return user_provided_key.strip()

    if GEMINI_API_KEY:
        return GEMINI_API_KEY

    return None

# ---------------------------------------------------------
# 5. System prompt bouwen
# ---------------------------------------------------------
def system_prompt():
    persona_json, description_json = load_profile()

    persona_session = st.session_state.get("persona", "")
    profile_session = st.session_state.get("profile", "")

    persona_block = "\n".join([persona_json, persona_session]).strip()
    description_block = "\n".join([description_json, profile_session]).strip()

    base_prompt = f"""
{persona_block}

{description_block}

Contextregels:
- Gebruik documentcontext wanneer relevant.
- Gebruik webcontext wanneer aanwezig.
- Als er geen context is, antwoord je vanuit je eigen redenering.
- Houd je aan de stijl, toon en identiteit zoals beschreven in het profiel.
- Vermijd overbodige herhaling, formele taal en technische disclaimers.
""".strip()

    return base_prompt

# ---------------------------------------------------------
# 6. Hoofdfunctie: Eva's antwoord
# ---------------------------------------------------------
def answer_question(question: str, context: str, use_document_index: bool = True):
    provider = st.session_state.get("ai_provider", "Gemini")
    st.session_state["debug_active_model"] = provider

    system_msg_content = system_prompt()
    chat_model_name = st.session_state.get("model_name", DEFAULT_MODEL_NAME)

    # -----------------------------
    # RAG context
    # -----------------------------
    document_context_str = ""
    if use_document_index:
        rag_mode = st.session_state.get("rag_mode", "auto")
        document_context_str = retrieve_context(question, mode=rag_mode)

    final_context = ""
    if document_context_str:
        final_context += f"Context uit documenten:\n{document_context_str}\n\n---\n\n"
    if context:
        final_context += f"Context uit geuploade bestanden:\n{context}\n\n---\n\n"
    if st.session_state.get("web_context"):
        final_context += f"Webcontext:\n{st.session_state['web_context']}\n\n---\n\n"
    if not final_context.strip():
        final_context = "(Geen extra context beschikbaar.)"

    # -----------------------------
    # Chatgeschiedenis
    # -----------------------------
    chat_history = []
    if "messages" in st.session_state and st.session_state.messages:
        for m in st.session_state.messages[-10:]:
            chat_history.append({"role": m["role"], "content": m["content"]})

    chat_ctx = ""
    if chat_history:
        chat_ctx = "\n".join(
            f"{c['role'].capitalize()}: {c['content']}"
            for c in chat_history
        )

    # -----------------------------
    # Prompt bouwen
    # -----------------------------
    prompt = (
        f"{system_msg_content}\n\n"
        f"{final_context}\n\n"
        f"{chat_ctx}\n\n"
        f"Vraag:\n{question}\n\nAntwoord:"
    )

    # -----------------------------
    # PROVIDER: GEMINI
    # -----------------------------
    if provider == "Gemini":
        try:
            gemini_api_key = get_gemini_api_key()
            if not gemini_api_key:
                return "[Geen Gemini API key gevonden]"

            client = genai.Client(api_key=gemini_api_key)

            response = client.models.generate_content(
                model=chat_model_name,
                contents=prompt,
                generation_config={
                    "temperature": st.session_state.get("temperature", DEFAULT_TEMPERATURE)
                }
            )

            answer = response.text.strip()
            return answer or "[Geen antwoord van Gemini]"

        except Exception as e:
            logging.exception("Gemini fout")
            return f"[FOUT bij Gemini: {e}]"

    # -----------------------------
    # PROVIDER: GROQ
    # -----------------------------
    if provider == "Groq":
        try:
            groq_key = st.secrets.get("GROQ_API_KEY", None)
            if not groq_key:
                return "[Geen Groq API key gevonden]"

            payload = {
                "model": "llama3-70b-8192",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": st.session_state.get("temperature", DEFAULT_TEMPERATURE)
            }

            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}"},
                json=payload,
                timeout=30
            )

            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            return answer or "[Geen antwoord van Groq]"

        except Exception as e:
            logging.exception("Groq fout")
            return f"[FOUT bij Groq: {e}]"

    return "[Onbekende provider]"

