# embeddings.py — Hugging Face compatibele versie
import numpy as np
import logging
import streamlit as st
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_DIMENSION

# ---------------------------------------------------------
# 1. Laad SentenceTransformer embedding model
# ---------------------------------------------------------
try:
    _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    _embedding_dim = _embedder.get_sentence_embedding_dimension()
except Exception as e:
    logging.exception(f"Kon embedding model niet laden: {e}")
    _embedder = None
    _embedding_dim = EMBEDDING_DIMENSION

# ---------------------------------------------------------
# 2. Embedding functie (cached)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_embedding_cached(text: str) -> np.ndarray:
    """
    Genereert een embedding via SentenceTransformers.
    Retourneert een numpy-array.
    Fallback: nulvector van EMBEDDING_DIMENSION.
    """
    if not text or not isinstance(text, str):
        return np.zeros(_embedding_dim, dtype=np.float32)

    if _embedder is None:
        logging.error("Embedding model niet beschikbaar — fallback naar nulvector.")
        return np.zeros(_embedding_dim, dtype=np.float32)

    try:
        emb = _embedder.encode(text, convert_to_numpy=True)
        if emb is None or emb.size == 0:
            raise ValueError("Lege embedding ontvangen.")
        return emb.astype(np.float32)
    except Exception as e:
        logging.exception(f"Embedding fout: {e}")
        return np.zeros(_embedding_dim, dtype=np.float32)

# ---------------------------------------------------------
# 3. Model capability check (optioneel)
# ---------------------------------------------------------
def model_supports_embeddings(model_name: str) -> bool:
    """
    Houdt deze functie aan voor compatibiliteit.
    In deze Hugging Face versie ondersteunt elk model embeddings via SentenceTransformers.
    """
    return True
