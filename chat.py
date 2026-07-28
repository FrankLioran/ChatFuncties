# chat.py
import logging
import json
from pathlib import Path

import streamlit as st
import requests

from documents import retrieve_context
from ai_router import ask_ai
from safety import check_limits

#from config import (
#    DEFAULT_MODEL_NAME,
#    DEFAULT_TEMPERATURE,
#    LOCAL_SECRETS_PATH,
#    PROFILE_FILENAME,
#)

# ---------------------------------------------------------
# ComfyUI functie 
# ---------------------------------------------------------

def generate_image_comfy(prompt: str):
    workflow = {
        "prompt": {
            "0": {
                "inputs": {
                    "text": prompt
                },
                "class_type": "CLIPTextEncode"
            },
            "1": {
                "inputs": {
                    "seed": 12345,
                    "steps": 20,
                    "cfg": 7,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": "checkpoint",
                    "positive": ["0"],
                    "negative": []
                },
                "class_type": "KSampler"
            },
            "2": {
                "inputs": {
                    "samples": ["1"],
                    "vae": "vae"
                },
                "class_type": "VAEDecode"
            },
            "3": {
                "inputs": {
                    "images": ["2"]
                },
                "class_type": "SaveImage"
            }
        }
    }

    response = requests.post("http://127.0.0.1:8188/prompt", json=workflow)
    data = response.json()

    # Haal de afbeelding op
    image_name = data["output"]["images"][0]["filename"]
    image_bytes = requests.get(f"http://127.0.0.1:8188/view?filename={image_name}").content

    return image_bytes
# ---------------------------------------------------------
# PROFIEL LADEN UIT eva_profile.json
# ---------------------------------------------------------

def get_persona_image():
    """
    Retourneert het pad naar de afbeelding die hoort bij de gekozen persona.
    """
    choice = st.session_state.get("active_persona", "Eva Lumen")

    image_map = {
        "Eva Lumen": "Eva.jpg",
        "Astraea": "Astraea.jpg",
        "Standaard": "default.jpg"
    }

    filename = image_map.get(choice, "default.jpg")
    return Path(__file__).parent / "images" / filename

def load_profile():
    """
    Laadt het profiel op basis van de persona‑switcher.
    """
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
# 3. SYSTEM PROMPT MET EVA'S PROFIEL
# ---------------------------------------------------------

def system_prompt():
    """
    Bouwt de system prompt op basis van:
    - Het gekozen persona-profiel (JSON)
    - Eventuele session_state overrides
    - Document- en webcontext
    """

    # 1. Laad profiel uit JSON
    persona_json, description_json = load_profile()

    # 2. Session overrides (optioneel)
    persona_session = st.session_state.get("persona", "")
    profile_session = st.session_state.get("profile", "")

    # 3. Combineer profielinformatie
    persona_block = "\n".join([persona_json, persona_session]).strip()
    description_block = "\n".join([description_json, profile_session]).strip()

    # 4. Bouw de system prompt
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
# 4. HOOFD FUNCTIE: EVA'S ANTWOORD
# ---------------------------------------------------------
def answer_question(
    question: str,
    context: str = "",
    use_document_index: bool = True):

    system_msg_content = system_prompt()

    rag_needed = (
        bool(st.session_state.get("sections")) or
        bool(st.session_state.get("document_index")) or
        bool(st.session_state.get("document_index_lazy"))
    )
    check_limits()
    document_context_str = ""

    if use_document_index and rag_needed:

        rag_mode = st.session_state.get("rag_mode", "auto")

        print(">>> retrieve_context wordt aangeroepen")

        document_context_str = retrieve_context(
            question,
            mode=rag_mode
        )

        print(">>> retrieve_context klaar")

        if document_context_str is None:
            document_context_str = ""

    # -------------------------------------------------
    # Context opbouwen
    # -------------------------------------------------

    final_context = ""

    if document_context_str:
        final_context += (
            "Context uit documenten:\n"
            f"{document_context_str}\n\n"
            "---\n\n"
        )

    if context:
        final_context += (
            "Context uit geüploade bestanden:\n"
            f"{context}\n\n"
            "---\n\n"
        )

    web_context = st.session_state.get("web_context", "")

    if web_context:
        final_context += (
            "Webcontext:\n"
            f"{web_context}\n\n"
            "---\n\n"
        )

    if not final_context:
        final_context = "(Geen extra context beschikbaar.)"

    # -------------------------------------------------
    # Chatgeschiedenis
    # -------------------------------------------------

    chat_history = []

    for m in st.session_state.get("messages", [])[-10:]:

        chat_history.append({
            "role": m["role"],
            # .get() voorkomt een crash als 'content' ontbreekt (bijv. bij een afbeelding)
            "content": m.get("content", "[Afbeelding]")
        })

    # -------------------------------------------------
    # Messages voor AI
    # -------------------------------------------------

    messages = []

    messages.append({
        "role": "system",
        "content": system_msg_content
    })

    if final_context:
        messages.append({
            "role": "system",
            "content": final_context
        })

    messages.extend(chat_history)

    messages.append({
        "role": "user",
        "content": question
    })

    # -------------------------------------------------
    # Debug
    # -------------------------------------------------

    #print("=" * 80)
    #print("MESSAGES NAAR AI")
    #print("=" * 80)

    #for i, m in enumerate(messages):
    #    print(f"\n[{i}] {m['role']}")
    #    print(m["content"][:300])

    #print("=" * 80)

    # -------------------------------------------------
    # AI
    # -------------------------------------------------

    try:
        return ask_ai(messages)

    except Exception as exc:
        logging.exception(exc)
        return f"[FOUT: {exc}]"
