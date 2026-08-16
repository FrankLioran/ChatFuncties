import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import requests
import streamlit as st

from ai_router import ask_ai
from documents import retrieve_context
from safety import check_limits


# ---------------------------------------------------------
# 1. COMFYUI AFBEELDING GENERATIE (Optioneel)
# ---------------------------------------------------------

def generate_image_comfy(prompt: str) -> bytes:
    """Genereert een afbeelding via een lokale ComfyUI instantie."""
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
    """Retourneert het pad naar de afbeelding die hoort bij de gekozen persona."""
    choice = st.session_state.get("active_persona", "Eva Lumen")

    image_map = {
        "Eva Lumen": "Eva.jpg",
        "Astraea": "Astraea.jpg",
        "Standaard": "default.jpg",
    }

    filename = image_map.get(choice, "default.jpg")
    return Path(__file__).parent / "images" / filename


@st.cache_data(show_spinner=False)
def load_profile_file(profile_path_str: str) -> Dict[str, Any]:
    """Leest een profielbestand éénmalig in via Streamlit caching."""
    path = Path(profile_path_str)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_profile() -> tuple[str, str]:
    """Laadt het persona-profiel op basis van de actieve selectie."""
    choice = st.session_state.get("active_persona", "Eva Lumen")

    profile_map = {
        "Eva Lumen": "eva_profile.json",
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
    """Bouwt de basale system prompt op uit profielbestanden en sessiedata."""
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
    """Ophaal-functie voor de gecachete system prompt."""
    persona_json, description_json = load_profile()

    return cached_system_prompt(
        persona_json,
        description_json,
        st.session_state.get("persona", ""),
        st.session_state.get("profile", ""),
    )


# ---------------------------------------------------------
# 4. HOOFDFUNCTIE: EVA'S ANTWOORD GENERATIE
# ---------------------------------------------------------

def answer_question(
    question: str, 
    context: str = "", 
    use_document_index: bool = True
) -> str:
    """
    Verwerkt de vraag van de gebruiker, verzamelt RAG-context,
    en stuurt een opgeruimd bericht naar het gekozen AI-model.
    """
    check_limits()

    # 1. Base system prompt ophalen
    base_system = system_prompt()

    # 2. Document- retrieval via RAG
    document_context_str = ""
    rag_needed = (
        bool(st.session_state.get("sections"))
        or bool(st.session_state.get("document_index"))
        or bool(st.session_state.get("document_index_lazy"))
    )

    if use_document_index and rag_needed:
        rag_mode = st.session_state.get("rag_mode", "auto")
        logging.info(f"Retrieval gestart met modus '{rag_mode}' voor vraag: {question[:50]}...")

        document_context_str = retrieve_context(question, mode=rag_mode) or ""

    # 3. Context-onderdelen verzamelen en bundelen
    context_parts = []

    if document_context_str.strip():
        context_parts.append(f"### DOCUMENT CONTEXT:\n{document_context_str}")

    if context.strip():
        context_parts.append(f"### DIRECTE BESTANDSCONTEXT:\n{context}")

    web_context = st.session_state.get("web_context", "")
    if web_context.strip():
        context_parts.append(f"### WEBCONTEXT:\n{web_context}")

    # 4. Punt 4 opgelost: Één gecombineerd System Block bouwen
    if context_parts:
        combined_context = "\n\n---\n\n".join(context_parts)
        full_system_prompt = f"{base_system}\n\n====================\nGEBRUIK DE ONDERSTAANDE CONTEXT OM DE VRAAG TE BEANTWOORDEN:\n\n{combined_context}\n===================="
    else:
        full_system_prompt = base_system

    # 5. Punt 3 & 5 opgelost: Schonere chatgeschiedenis-slice zonder dubbele vraag
    raw_messages = st.session_state.get("messages", [])

    # Als de meest recente message in session_state al de vraag van de gebruiker is, 
    # sluiten we deze uit van de historie om dubbele verzending te voorkomen.
    if raw_messages and raw_messages[-1].get("role") == "user" and raw_messages[-1].get("content") == question:
        history_source = raw_messages[:-1]
    else:
        history_source = raw_messages

    # Neem de laatste 10 geschiedenisberichten en filter speciale/invalid rollen
    formatted_history = []
    for m in history_source[-10:]:
        role = m.get("role", "user")

        # Mappen van speciale rollen naar geaccepteerde API rollen
        if role == "image":
            role = "assistant"
            content = m.get("content", "🎨 [Afbeelding gegenereerd]")
        else:
            content = m.get("content", "")

        if role in ("user", "assistant", "system") and content:
            formatted_history.append({"role": role, "content": content})

    # 6. Samenstellen van de definitieve berichtenlijst
    messages: List[Dict[str, str]] = []

    # A. Het gecombineerde system block
    messages.append({"role": "system", "content": full_system_prompt})

    # B. De opgeruimde chatgeschiedenis
    messages.extend(formatted_history)

    # C. De actuele vraag (exact 1 keer aan het einde)
    messages.append({"role": "user", "content": question})

    # Debug-logging voor de console
    logging.info(f"Totaal aantal berichten naar AI: {len(messages)} (Geschiedenis: {len(formatted_history)})")

    # 7. Aanpakken van het AI-router model
    try:
        return ask_ai(messages)
    except Exception as exc:
        logging.exception(f"Fout tijdens ask_ai aanroep: {exc}")
        return f"🛑 Er is een fout opgetreden bij het verwerken van je vraag: {exc}"
