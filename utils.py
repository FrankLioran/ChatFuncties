# utils.py — Hugging Face compatibele versie
import numpy as np
import json
from datetime import datetime, timezone
from pathlib import Path
import logging
from config import LOG_DIR

# ---------------------------------------------------------
# 1. Tekst splitting (consistent met documents.py)
# ---------------------------------------------------------
def split_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list:
    """
    Splits tekst in overlappende chunks.
    Compatibel met Hugging Face.
    """
    if not isinstance(text, str):
        return []

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    L = len(text)

    while start < L:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(0, end - overlap)

    return chunks

# ---------------------------------------------------------
# 2. Cosine similarity
# ---------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity tussen twee vectors.
    Robuust tegen lege of None embeddings.
    """
    if a is None or b is None:
        return 0.0

    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        return 0.0

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0 or nb == 0:
        return 0.0

    return float(np.dot(a, b) / (na * nb))

# ---------------------------------------------------------
# 3. Logging (audit events)
# ---------------------------------------------------------
def log_event(event_type: str, details: dict):
    """
    Schrijft een auditlog naar LOG_DIR/eva_audit.log.
    Werkt volledig op Hugging Face.
    """
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / "eva_audit.log"

        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **details
        }

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    except Exception as e:
        logging.exception(f"Kon auditlog niet schrijven: {e}")

# utils.py
FREE_TOKEN_BUDGET = 400_000   # je eigen safe‑margin
if st.session_state.get("used_tokens", 0) + new_prompt_tokens > FREE_TOKEN_BUDGET:
    st.warning("We naderen de gratis limiet. Ik zal nu korter antwoorden.")
    temperature = 0.5
    max_tokens = 300

