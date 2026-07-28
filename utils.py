# utils.py
import numpy as np
import json
from datetime import datetime, timezone
from pathlib import Path
import logging
import streamlit as st
from config import LOG_DIR, CHAT_SAVE_DIR

def split_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list:
    chunks = []
    start = 0
    L = len(text)
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk_size")
    while start < L:
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = max(0, end - overlap)
    return [c for c in chunks if c]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def log_event(event_type: str, details: dict):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "eva_audit.log"
    entry = {"ts": datetime.now(timezone.utc).isoformat(), "event": event_type, **details}
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        logging.exception("Kon auditlog niet schrijven")

# --- Hulpfunctie om chat op te slaan ---
def save_chat_to_txt(messages: list) -> Path | None:

    if not messages:
        return None

    try:

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = CHAT_SAVE_DIR / f"eva_chat_{ts}.txt"

        with open(fp, "w", encoding="utf-8") as f:

            f.write("Eva Lumen Chat\n")
            f.write(f"Datum: {datetime.now()}\n")
            f.write(f"Provider: {st.session_state.get('ai_provider')}\n")
            f.write(f"Model: {st.session_state.get('model_name')}\n")
            f.write(f"Persona: {st.session_state.get('active_persona')}\n")
            f.write("=" * 60 + "\n\n")

            for m in messages:
                f.write(f"{m['role'].upper()}:\n")
                f.write(f"{m['content']}\n\n")

        return fp

    except Exception:
        logging.exception("Kon gesprek niet opslaan")
        return None
