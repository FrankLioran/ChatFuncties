# memory.py
import streamlit as st
from pathlib import Path
from datetime import datetime
from utils import split_text
from documents import retrieve_context
from ai_router import ask_ollama

MEMORY_FOLDER = Path("memory_logs")
MEMORY_FOLDER.mkdir(exist_ok=True)

def get_memory_mode() -> str:
    return st.session_state.get("memory_mode", "summary_rag")

def _summary_prompt(history_text: str):
    return [
        {
            "role": "system",
            "content": (
                "Vat het gesprek samen in maximaal 10 zinnen. "
                "Focus op intenties, voorkeuren, beslissingen en openstaande vragen. "
                "Geen meta-commentaar, geen excuses."
            ),
        },
        {"role": "user", "content": history_text},
    ]

def update_conversation_summary(messages):
    mode = get_memory_mode()
    if mode not in ("summary", "summary_rag"):
        return

    lines = []
    for m in messages:
        if m["role"] in ("user", "assistant"):
            lines.append(f"{m['role'].upper()}: {m['content']}")

    history_text = "\n".join(lines)[-8000:]

    if not history_text.strip():
        return

    try:
        summary = ask_ollama(_summary_prompt(history_text))
        st.session_state["conversation_summary"] = summary.strip()
    except Exception as e:
        st.session_state["conversation_summary"] = f"[Samenvatting mislukt: {e}]"

def save_conversation_to_memory(messages):
    mode = get_memory_mode()
    if mode != "summary_rag":
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = st.session_state.get("session_id", "unknown")

    fp = MEMORY_FOLDER / f"chat_{timestamp}_{session_id}.txt"

    lines = []
    for m in messages:
        if m["role"] in ("user", "assistant"):
            lines.append(f"{m['role'].upper()}: {m['content']}")

    text = "\n".join(lines)
    fp.write_text(text, encoding="utf-8")

    chunks = split_text(text, chunk_size=800, overlap=100)
    st.session_state["memory_chunks"] = chunks

    return fp

def retrieve_memory_context(question: str) -> str:
    mode = get_memory_mode()
    if mode != "summary_rag":
        return ""

    chunks = st.session_state.get("memory_chunks", [])
    if not chunks:
        return ""

    original_sections = st.session_state.get("sections", [])

    try:
        st.session_state["sections"] = chunks
        ctx = retrieve_context(question, mode="semantic")
    finally:
        st.session_state["sections"] = original_sections

    return ctx or ""
