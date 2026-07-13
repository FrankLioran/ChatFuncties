# app.py — Hugging Face compatibele versie
import streamlit as st
import uuid
import random
import os
import json
import google.genai as genai

from pathlib import Path

# Projectmodules
from config import (
    DOCUMENT_FOLDER,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    GEMINI_API_KEY
)

from chat import answer_question, load_profile, get_persona_image
from documents import scan_and_index_folder_full, scan_and_index_folder_lazy

# ---------------------------------------------------------
# 1. Audio uitschakelen (werkt niet op Hugging Face)
# ---------------------------------------------------------
def listen_and_transcribe():
    st.warning("Audio-invoer wordt niet ondersteund op Hugging Face Spaces.")
    return ""

# ---------------------------------------------------------
# 2. ComfyUI / Paint.NET uitschakelen (werkt niet op Hugging Face)
# ---------------------------------------------------------
def start_comfyui(): 
    st.warning("ComfyUI kan niet worden gestart op Hugging Face Spaces.")

def start_paintnet():
    st.warning("Paint.NET kan niet worden gestart op Hugging Face Spaces.")

def generate_comfy_image(*args, **kwargs):
    st.warning("Lokale ComfyUI beeldgeneratie is niet beschikbaar op Hugging Face.")
    return None

def comfyui_generate_video_from_image(*args, **kwargs):
    st.warning("Video-generatie via ComfyUI is niet beschikbaar op Hugging Face.")
    return None

# ---------------------------------------------------------
# 3. Pollinations (werkt wél op Hugging Face)
# ---------------------------------------------------------
from image import (
    generate_pollinations,
    generate_pollinations_styled,
    save_pollinations_image,
    build_pollinations_url,
    fetch_pollinations_image,
    save_to_comfyui_input,
    generate_pollinations_image,
    upload_image_for_pollinations,
    poll_style_flux,
    poll_style_sdxl,
    poll_style_anime,
    poll_style_realistic,
    poll_style_cinematic,
    poll_ar_square,
    poll_ar_wide,
    poll_ar_vertical,
    poll_negative_no_text,
    poll_negative_no_blur,
    poll_seed_random,
    poll_seed_fixed,
    poll_steps_fast,
    poll_steps_quality,
    poll_image_to_image
)

# ---------------------------------------------------------
# 4. Streamlit configuratie
# ---------------------------------------------------------
st.set_page_config(page_title="Eva — Vraag & Antwoord", layout="wide")

# ---------------------------------------------------------
# 6. Sessiestatus initialiseren
# ---------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []
    def get_welcome_line(temp=0.7):
        opts = [
            "Welkom 🌿 — ik ben Eva. Waarmee kan ik helpen?",
            "Hoi — ik help je graag rustig en helder.",
            "Hallo — vraag maar, ik denk met je mee."
        ]
        return random.choice(opts)
    st.session_state.messages.append({"role": "assistant", "content": get_welcome_line(DEFAULT_TEMPERATURE)})

defaults = {
    "model_name": DEFAULT_MODEL_NAME,
    "temperature": DEFAULT_TEMPERATURE,
    "document_folder": str(DOCUMENT_FOLDER),
    "document_index": [],
    "pdf_text": None,
    "rag_mode": "auto",
    "sections": [],
    "section_embeddings": [],
    "spoken_text_input": "",
    "spoken_processed": False,
    "persona": None,
    "profile": None,
    "excel_df": None,
    "web_context": "",
    "original_image": None,
    "current_image": None,
    "image_config": None,
    "uploaded_paint_file": None,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------
# 7. Persona laden
# ---------------------------------------------------------
persona_content, profile_text = load_profile()
st.session_state.persona = persona_content
st.session_state.profile = profile_text

# ---------------------------------------------------------
# 8. Titel
# ---------------------------------------------------------
st.title("Eva — Vraag & Antwoord")

# ---------------------------------------------------------
# 9. Sidebar
# ---------------------------------------------------------
with st.sidebar:

    # Persona afbeelding
    st.image(str(get_persona_image()), width=220)

    # Persona keuze
    st.header("Persona")
    persona_choice = st.selectbox(
        "Kies een persona",
        ["Eva Lumen", "Astraea", "Standaard"],
        index=0
    )
    st.session_state["active_persona"] = persona_choice

    # Profiel opnieuw laden
    persona_content, profile_text = load_profile()
    st.session_state.persona = persona_content
    st.session_state.profile = profile_text

    # Modelinstellingen
    st.header("Modelinstellingen")
    st.session_state.model_name = st.selectbox(
        "Modelnaam",
        [
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash"
        ],
        index=0
    )
    
    st.header("AI Provider")
    st.session_state["ai_provider"] = st.selectbox(
        "Kies een AI-provider",
        ["Gemini", "Groq"],
        index=0
    )

    if st.session_state["ai_provider"] == "Groq":
    st.session_state["groq_model"] = st.selectbox(
        "Groq model",
        ["llama3-70b-8192", "llama3-8b-8192", "gemma2-9b-it", "mixtral-8x7b-32768"],
        index=0
    )

    # Temperature
    st.session_state.temperature = st.slider(
        "Temperature",
        0.0,
        1.5,
        st.session_state.temperature
    )

    # Document Indexering
    st.header("📁 Document Indexering")

    folder = st.session_state.get("document_folder")

    if st.button("🔍 Full index (RAG)"):
        scan_and_index_folder_full(folder)

    if st.button("⚡ Lazy index (samenvattingen)"):
        scan_and_index_folder_lazy(folder)

    if st.button("🤖 Hybrid index (auto RAG)"):
        st.session_state["rag_mode"] = "hybrid"
        st.success("Hybrid RAG-modus geactiveerd.")

    if st.button("🗑 Index wissen"):
        st.session_state.document_index = []
        st.session_state.document_index_lazy = []

        folder = Path(st.session_state["document_folder"])
        index_file = folder / "document_index.json"
        index_file_lazy = folder / "document_index_lazy.json"

        if index_file.exists():
            index_file.unlink()
        if index_file_lazy.exists():
            index_file_lazy.unlink()

        st.success("Indexbestanden verwijderd.")

    st.subheader("📊 Index status")
    st.write(f"Full index: **{len(st.session_state.get('document_index', []))} chunks**")
    st.write(f"Lazy index: **{len(st.session_state.get('document_index_lazy', []))} documenten**")

# ---------------------------------------------------------
# 10. Chat input
# ---------------------------------------------------------
user_input = st.chat_input("Typ je vraag…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    answer = answer_question(user_input, context="", use_document_index=True)

    if isinstance(answer, bytes):
        st.session_state.messages.append({"role": "assistant", "content": "[[IMAGE_RESULT]]"})
        with st.chat_message("assistant"):
            st.image(answer)
    else:
        st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------------------------------------------------------
# 11. Chatweergave
# ---------------------------------------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
