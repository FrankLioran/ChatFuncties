import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import requests
import streamlit as st

from ai_router import ask_ai
from documents import retrieve_context
from safety import check_limits
from memory import (
    update_conversation_summary,
    save_conversation_to_memory,
    retrieve_memory_context,
    get_memory_mode,
)

# ---------------------------------------------------------
# 1. COMFYUI AFBEELDING GENERATIE (Optioneel)
# ---------------------------------------------------------

def generate_image_comfy(prompt: str) -> bytes:
    workflow = {
        "prompt": {
            "0": {"inputs": {"text": prompt}, "class_type": "CLIPTextEncode"},
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
                    "negative": [],
                },
                "class_type": "KSampler",
            },
            "2": {
                "inputs": {"samples": ["1"], "vae": "vae"},
                "class_type": "VAEDecode",
            },
            "3": {"inputs": {"images": ["2"]}, "class_type": "SaveImage"},
        }
    }

    try:
        response = requests.post("http://127.0.0.1:8188/prompt", json=workflow, timeout=30)
        response.raise_for_status()
        data = response.json()

        image_name = data["output"]["images"][0]["filename"]
        img_response = requests.get(
            f"http://127.0.0.1:8188/view?filename={image_name}", timeout=30
        )
        img_response.raise_for_status()
        return img_response.content
    except Exception as e:
        logging.exception(f"ComfyUI generatie mislukt: {e}")
        raise RuntimeError(f"ComfyUI niet bereikbaar: {e}")


# ---------------------------------------------------------
# 2. PERSONA & PROFIEL BEHEER
# ---------------------------------------------------------

def get_persona_image() -> Path:
    choice = st.session_state.get("active_persona", "Eva Lumen")

    image_map = {
        "Eva Lumen": "Eva.jpg",
        "Astraea": "Astraea.jpg",
        "Helion Arcturus": "Helion.jpg",
        "Standaard": "default.jpg",
    }

    filename = image_map.get(choice, "default.jpg")
    return Path(__file__).parent / "images" / filename


@st.cache_data(show_spinner=False)
def load_profile_file(profile_path_str: str) -> Dict[str, Any]:
    path = Path(profile_path_str)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def trim_text_tokens(text: str, max_tokens: int = 2000) -> str:
    """
    Trim text to a maximum token estimate.
    1 token ≈ 4 chars (ruwe schatting).
    """
    if not text:
        return ""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[TRIMMED]..."

def load_profile() -> tuple[str, str]:
    choice = st.session_state.get("active_persona", "Eva Lumen")

    profile_map = {
        "Eva Lumen": "eva_profile.json",
        "Helion Arcturus": "helion_arcturus.json",
        "Astraea": "astraea_profile.json",
        "Standaard": "default_profile.json",
    }

    filename = profile_map.get(choice, "default_profile.json")
    profile_path = Path(__file__).parent / "profiles" / filename

    if not profile_path.exists():
        logging.warning(f"Profielbestand niet gevonden: {profile_path}")
        return "", ""

    try:
        data = load_profile_file(str(profile_path))
        return (data.get("persona", ""), data.get("description", ""))
    except Exception as e:
        logging.exception(f"Kon profiel niet laden: {e}")
        return "", ""


# ---------------------------------------------------------
# 3. SYSTEM PROMPT OPBOUW
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_system_prompt(
    persona_json: str,
    description_json: str,
    persona_session: str,
    profile_session: str,
) -> str:
    persona_block = "\n".join(filter(None, [persona_json, persona_session])).strip()
    description_block = "\n".join(filter(None, [description_json, profile_session])).strip()

    return f"""
{persona_block}

{description_block}

Contextregels:
- Gebruik documentcontext wanneer relevant.
- Gebruik webcontext wanneer aanwezig.
- Als er geen context is, antwoord je vanuit je eigen redenering.
- Houd je aan de stijl, toon en identiteit zoals beschreven in het profiel.
- Vermijd overbodige herhaling, formele taal en technische disclaimers.
""".strip()


def system_prompt() -> str:
    persona_json, description_json = load_profile()

    return cached_system_prompt(
        persona_json,
        description_json,
        st.session_state.get("persona", ""),
        st.session_state.get("profile", ""),
    )


# ---------------------------------------------------------
# 4. MEMORY + RAG + TOKEN TRIM ENGINE
# ---------------------------------------------------------

def trim_messages_to_token_limit(messages, max_tokens=3000):
    """
    Slimme token-trim:
    - Houdt system prompt altijd
    - Houdt laatste user-vraag altijd
    - Verwijdert oudste assistant/user berichten
    - Schatting: 1 token ≈ 4 chars
    """
    def estimate_tokens(msgs):
        return sum(len(m.get("content", "")) // 4 for m in msgs)

    tokens = estimate_tokens(messages)

    while tokens > max_tokens and len(messages) > 2:
        # verwijder oudste niet-system bericht
        messages.pop(1)
        tokens = estimate_tokens(messages)

    return messages

def answer_question(question: str, context: str = "", use_document_index: bool = True, stream: bool = False) -> str:
    check_limits()

    base_system = system_prompt()

    # 1. Document-RAG
    document_context_str = ""
    rag_needed = (
        bool(st.session_state.get("sections"))
        or bool(st.session_state.get("document_index"))
        or bool(st.session_state.get("document_index_lazy"))
    )

    # Context ophalen kan enkele seconden duren
    with st.spinner("🧠 Context verzamelen…"):
        if use_document_index and rag_needed:
            rag_mode = st.session_state.get("rag_mode", "auto")
            document_context_str = retrieve_context(question, mode=rag_mode) or ""

        # 2. Memory
        memory_mode = get_memory_mode()
        summary = st.session_state.get("conversation_summary", "")
        memory_ctx = (
            retrieve_memory_context(question)
            if memory_mode == "summary_rag"
            else ""
        )
        web_context = st.session_state.get("web_context", "")

    provider = st.session_state.get("ai_provider", "Lokaal")

    # 3. Context bundelen
    context_parts = []

    # Trim afhankelijk van provider
    if provider == "Gemini":
        summary_trimmed = summary
        memory_trimmed = memory_ctx
        doc_trimmed = document_context_str
        direct_trimmed = context
        web_trimmed = web_context
    else:
        summary_trimmed = trim_text_tokens(summary, max_tokens=2000)
        memory_trimmed = trim_text_tokens(memory_ctx, max_tokens=2000)
        doc_trimmed = trim_text_tokens(document_context_str, max_tokens=2000)
        direct_trimmed = trim_text_tokens(context, max_tokens=2000)
        web_trimmed = trim_text_tokens(web_context, max_tokens=2000)

    # Nu pas toevoegen aan context_parts
    if summary_trimmed.strip():
        context_parts.append(f"### GESPREKSSAMENVATTING:\n{summary_trimmed}")

    if memory_trimmed.strip():
        context_parts.append(f"### VORIGE GESPREKSCONTEXT (RAG):\n{memory_trimmed}")

    if doc_trimmed.strip():
        context_parts.append(f"### DOCUMENT CONTEXT:\n{doc_trimmed}")

    if direct_trimmed.strip():
        context_parts.append(f"### DIRECTE BESTANDSCONTEXT:\n{direct_trimmed}")

    if web_trimmed.strip():
        context_parts.append(f"### WEBCONTEXT:\n{web_trimmed}")


    if context_parts:
        if provider == "Gemini":
            combined_context = "\n\n---\n\n".join(context_parts)
        else:
            combined_context = trim_text_tokens("\n\n---\n\n".join(context_parts), max_tokens=3000)

        full_system_prompt = (
            f"{base_system}\n\n"
            "====================\n"
            "GEBRUIK DE ONDERSTAANDE CONTEXT OM DE VRAAG TE BEANTWOORDEN:\n\n"
            f"{combined_context}\n"
            "===================="
        )
    else:
        full_system_prompt = base_system

    # 4. Chatgeschiedenis
    raw_messages = st.session_state.get("messages", [])
    if raw_messages and raw_messages[-1].get("role") == "user" and raw_messages[-1].get("content") == question:
        history = raw_messages[:-1]
    else:
        history = raw_messages

    messages = [{"role": "system", "content": full_system_prompt}]

    for m in history[-4:]:
        role = m.get("role", "")
        content = m.get("content", "")

        if role in ("user", "assistant"):
            messages.append({"role": role, "content": str(content)})

    messages.append({"role": "user", "content": question})

    # 5. Token-trim

    provider = st.session_state.get("ai_provider", "Lokaal")

    if provider == "Lokaal":
        # lokale modellen hebben kleine context → trimmen
        messages = trim_messages_to_token_limit(messages, max_tokens=3000)

    elif provider == "Groq":
        # Groq heeft ±32k context → lichte trim
        messages = trim_messages_to_token_limit(messages, max_tokens=8000)

    elif provider == "Gemini":
        # Gemini heeft 128k–2M context → NIET trimmen
        pass

    # STREAMING voor lokale modellen
    if stream and st.session_state.get("ai_provider") == "Lokaal":
        from ai_router import stream_ai
        return stream_ai(messages)   # generator teruggeven

    # 6. Vraag aan model
    answer = ask_ai(messages)

    # 7. Geheugen bijwerken
    # Alleen opslaan als het GEEN streaming is
    if not stream:
        st.session_state.messages.append({"role": "assistant", "content": answer})
        update_conversation_summary(st.session_state.messages)
        save_conversation_to_memory(st.session_state.messages)

    return answer