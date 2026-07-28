# embeddings.py
# ---------------------------------------------------------
# EMBEDDING PROVIDER & VECTOR GENERATION SYSTEM
# ---------------------------------------------------------

import logging
import os
import requests
import numpy as np
import streamlit as st
import ollama
from typing import Optional

from config import EMBEDDING_DIMENSION

# ---------------------------------------------------------
# 1. HELPER FUNCTIES & CACHED PROVIDER CHECKS
# ---------------------------------------------------------

@st.cache_resource(ttl=300)
def ollama_available() -> bool:
    """
    Controleert of de lokale Ollama instantie bereikbaar is.
    Gecachet om herhaaldelijke netwerk-pings tijdens chunking te voorkomen.
    """
    try:
        ollama.list()
        return True
    except Exception:
        return False


def get_gemini_api_key() -> Optional[str]:
    """
    Ophaalt de Gemini API-sleutel volgens de juiste prioriteit:
    1. Streamlit Session State
    2. Streamlit Secrets
    3. Omgevingsvariabelen (os.environ)
    """
    if "gemini_api_key_user" in st.session_state and st.session_state.gemini_api_key_user:
        return st.session_state.gemini_api_key_user

    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    return os.environ.get("GEMINI_API_KEY")


def get_gemini_embedding(text: str, api_key: str) -> Optional[np.ndarray]:
    """
    Haalt embeddings op via de officiële Gemini API (text-embedding-004).
    """
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "content": {
                "parts": [{"text": text}]
            }
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()

        data = resp.json()
        embedding_values = data.get("embedding", {}).get("values")
        if embedding_values and isinstance(embedding_values, list):
            return np.array(embedding_values, dtype=np.float32)
        else:
            logging.warning(f"Onverwachte response structuur van Gemini API: {data}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Netwerkfout bij Gemini embedding API: {e}")
    except Exception as e:
        logging.error(f"Fout bij ophalen Gemini embedding: {e}")

    return None


# ---------------------------------------------------------
# 2. MAIN CACHED EMBEDDING FUNCTION
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_embedding_cached(text: str, model: str = "mxbai-embed-large:latest") -> np.ndarray:
    """
    Centrale functie voor het genereren van embeddings.
    Probeert achtereenvolgens:
    1. Lokale Ollama service (`ollama.embed`)
    2. Cloud Gemini API (`text-embedding-004`)
    3. Fallback naar Nulvector.
    """
    clean_text = text.strip() if text else ""
    if not clean_text:
        return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)

    # 1. Probeer lokaal via Ollama
    if ollama_available():
        try:
            resp = ollama.embed(model=model, input=clean_text)
            embedding_list = None

            if isinstance(resp, dict):
                if "embeddings" in resp and resp["embeddings"]:
                    embedding_list = resp["embeddings"][0]
                elif "embedding" in resp and isinstance(resp["embedding"], list):
                    embedding_list = resp["embedding"]
            elif hasattr(resp, "embeddings") and resp.embeddings:
                embedding_list = resp.embeddings[0]

            if embedding_list:
                arr = np.array(embedding_list, dtype=np.float32).flatten()
                if arr.shape[0] == EMBEDDING_DIMENSION:
                    logging.debug(f"Embedding succesvol gegenereerd via Ollama ({model}).")
                    return arr
                else:
                    logging.warning(
                        f"Ollama vector dimensie ({arr.shape[0]}) wijkt af van verwachte {EMBEDDING_DIMENSION}."
                    )
                    return arr
        except Exception as e:
            logging.warning(f"Lokale Ollama embedding mislukt, fallback naar cloud... ({e})")

    # 2. Cloud Fallback: Gemini
    gemini_key = get_gemini_api_key()
    if gemini_key:
        emb = get_gemini_embedding(clean_text, gemini_key)
        if emb is not None:
            if emb.shape[0] == EMBEDDING_DIMENSION:
                logging.debug("Embedding succesvol gegenereerd via Gemini API.")
                return emb
            else:
                logging.warning(
                    f"Gemini vector dimensie ({emb.shape[0]}) wijkt af van verwachte {EMBEDDING_DIMENSION}."
                )
                return emb

    # 3. Uiterste fallback: Nulvector
    logging.warning("Geen actieve embedding provider beschikbaar. Fallback naar nulvector.")
    return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)


# ---------------------------------------------------------
# 3. UTILITY / COMPATIBILITY CHECKS
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def model_supports_embeddings(model_name: str) -> bool:
    """
    Controleert of een modelnaam bekend staat om embedding-ondersteuning.
    """
    if not model_name:
        return False
    nm = model_name.lower()
    known_keywords = [
        "embed", "embedding", "nomic", "mxbai", "arctic",
        "bge", "gte", "e5", "snowflake", "jina", "voyage",
        "all-minilm", "text-embedding", "multilingual-e5"
    ]
    return any(kw in nm for kw in known_keywords)
