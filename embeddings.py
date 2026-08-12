# embeddings.py

import logging
import os
from typing import Optional

import numpy as np
import requests
import streamlit as st
import ollama

import config


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# 1. CONFIGURATIE
# ---------------------------------------------------------

EMBEDDING_MODEL = getattr(
    config,
    "EMBEDDING_MODEL",
    "mxbai-embed-large:latest",
)

GEMINI_EMBEDDING_MODEL = getattr(
    config,
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001",
)


# ---------------------------------------------------------
# 2. OLLAMA BESCHIKBAARHEID
# ---------------------------------------------------------

@st.cache_resource(ttl=300)
def ollama_available() -> bool:
    """
    Controleert of de lokale Ollama-service bereikbaar is.

    Deze functie controleert alleen de bereikbaarheid van Ollama.
    Er wordt hier bewust NIET gecontroleerd of een specifiek
    embedding-model beschikbaar is.

    Returns:
        True wanneer Ollama bereikbaar is.
        False wanneer de verbinding mislukt.
    """
    try:
        ollama.list()
        return True

    except Exception as exc:
        logger.debug(
            "Ollama is niet beschikbaar: %s",
            exc,
        )
        return False


# ---------------------------------------------------------
# 3. GEMINI API KEY
# ---------------------------------------------------------

def get_gemini_api_key() -> Optional[str]:
    """
    Haalt de Gemini API-key op.

    Prioriteit:
    1. Streamlit session state
    2. Streamlit secrets
    3. Omgevingsvariabele

    Returns:
        API-key als deze beschikbaar is, anders None.
    """

    # 1. Session state
    try:
        session_key = st.session_state.get("gemini_api_key_user")

        if session_key:
            return str(session_key).strip()

    except Exception:
        pass

    # 2. Streamlit secrets
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY")

        if secret_key:
            return str(secret_key).strip()

    except Exception:
        pass

    # 3. Environment
    env_key = os.environ.get("GEMINI_API_KEY")

    if env_key:
        return env_key.strip()

    return None


# ---------------------------------------------------------
# 4. EMBEDDING VALIDATIE
# ---------------------------------------------------------

def _validate_embedding(
    embedding,
    provider: str,
) -> Optional[np.ndarray]:
    """
    Zet een embedding om naar een nette float32 numpy-array.

    Er wordt bewust GEEN vaste dimensie gecontroleerd.

    De dimensie wordt bepaald door het gebruikte embedding-model.

    Args:
        embedding: Ruwe embedding-data.
        provider: Naam van de provider voor logging.

    Returns:
        1D numpy-array met dtype float32, of None bij een ongeldige
        embedding.
    """

    if embedding is None:
        logger.warning(
            "%s gaf geen embedding terug.",
            provider,
        )
        return None

    try:
        vector = np.asarray(
            embedding,
            dtype=np.float32,
        ).flatten()

    except (TypeError, ValueError) as exc:
        logger.error(
            "%s gaf een embedding terug die niet naar "
            "numpy.float32 kan worden geconverteerd: %s",
            provider,
            exc,
        )
        return None

    if vector.size == 0:
        logger.warning(
            "%s gaf een lege embedding terug.",
            provider,
        )
        return None

    if not np.all(np.isfinite(vector)):
        logger.warning(
            "%s gaf een embedding met NaN/Inf-waarden terug.",
            provider,
        )
        return None

    return vector


# ---------------------------------------------------------
# 5. OLLAMA EMBEDDING
# ---------------------------------------------------------

def get_ollama_embedding(
    text: str,
    model: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Genereert lokaal een embedding via Ollama.

    Args:
        text: Tekst waarvoor een embedding moet worden gemaakt.
        model: Optioneel Ollama-model. Indien None wordt
               EMBEDDING_MODEL gebruikt.

    Returns:
        numpy-array met embedding of None bij een fout.
    """

    clean_text = text.strip() if text else ""

    if not clean_text:
        logger.debug(
            "Geen embedding gegenereerd: lege tekst."
        )
        return None

    model_name = model or EMBEDDING_MODEL

    if not ollama_available():
        return None

    try:
        response = ollama.embed(
            model=model_name,
            input=clean_text,
        )

    except Exception as exc:
        logger.warning(
            "Ollama embedding mislukt voor model '%s': %s",
            model_name,
            exc,
        )
        return None

    # -----------------------------------------------------
    # Moderne Ollama response:
    #
    # {
    #     "embeddings": [[...]]
    # }
    # -----------------------------------------------------

    embedding_data = None

    if isinstance(response, dict):

        embeddings = response.get("embeddings")

        if isinstance(embeddings, list) and embeddings:
            embedding_data = embeddings[0]

        # -------------------------------------------------
        # Compatibiliteit met oudere response:
        #
        # {
        #     "embedding": [...]
        # }
        # -------------------------------------------------

        elif isinstance(response.get("embedding"), list):
            embedding_data = response.get("embedding")

    else:
        # Compatibiliteit met response-objecten
        try:
            embeddings = getattr(
                response,
                "embeddings",
                None,
            )

            if embeddings:
                embedding_data = embeddings[0]

            else:
                embedding_data = getattr(
                    response,
                    "embedding",
                    None,
                )

        except Exception:
            embedding_data = None

    vector = _validate_embedding(
        embedding_data,
        provider=f"Ollama/{model_name}",
    )

    if vector is None:
        logger.warning(
            "Ollama leverde geen geldige embedding voor model '%s'.",
            model_name,
        )
        return None

    logger.debug(
        "Ollama embedding succesvol: model=%s, dimensie=%d",
        model_name,
        vector.shape[0],
    )

    return vector


# ---------------------------------------------------------
# 6. GEMINI EMBEDDING
# ---------------------------------------------------------

def get_gemini_embedding(
    text: str,
    api_key: str,
    model: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Genereert een embedding via de Gemini API.

    Args:
        text: Tekst waarvoor een embedding moet worden gemaakt.
        api_key: Gemini API-key.
        model: Optioneel Gemini embedding-model.

    Returns:
        numpy-array met embedding of None bij een fout.
    """

    clean_text = text.strip() if text else ""

    if not clean_text:
        return None

    if not api_key:
        return None

    model_name = model or GEMINI_EMBEDDING_MODEL

    # Gemini API gebruikt modelnamen zonder "models/" in het pad.
    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model_name}:embedContent"
    )

    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "content": {
            "parts": [
                {
                    "text": clean_text,
                }
            ]
        }
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            params={"key": api_key},
            json=payload,
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

    except requests.exceptions.Timeout:
        logger.error(
            "Timeout bij Gemini embedding API."
        )
        return None

    except requests.exceptions.RequestException as exc:
        logger.error(
            "Netwerkfout bij Gemini embedding API: %s",
            exc,
        )
        return None

    except ValueError as exc:
        logger.error(
            "Gemini gaf geen geldige JSON-response: %s",
            exc,
        )
        return None

    # Verwachte structuur:
    #
    # {
    #   "embedding": {
    #       "values": [...]
    #   }
    # }

    embedding_data = (
        data
        .get("embedding", {})
        .get("values")
    )

    vector = _validate_embedding(
        embedding_data,
        provider=f"Gemini/{model_name}",
    )

    if vector is None:
        logger.warning(
            "Gemini leverde geen geldige embedding."
        )
        return None

    logger.debug(
        "Gemini embedding succesvol: model=%s, dimensie=%d",
        model_name,
        vector.shape[0],
    )

    return vector


# ---------------------------------------------------------
# 7. CENTRALE EMBEDDING FUNCTIE
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_embedding_cached(
    text: str,
    model: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Centrale functie voor embedding-generatie.

    Provider-volgorde:
        1. Ollama
        2. Gemini

    Belangrijk:
        Deze functie retourneert NOOIT een nulvector.

        Bij een echte embedding-fout wordt None teruggegeven.
        Hierdoor kan de retrieval-laag herkennen dat embeddings
        niet beschikbaar zijn en kan zij transparant besluiten
        wat zij moet doen.

    Args:
        text:
            Tekst waarvoor een embedding moet worden gemaakt.

        model:
            Optioneel Ollama-model. Als None wordt het centraal
            geconfigureerde EMBEDDING_MODEL gebruikt.

    Returns:
        numpy.ndarray met dynamische dimensie,
        of None wanneer geen provider beschikbaar is.
    """

    clean_text = text.strip() if text else ""

    if not clean_text:
        return None

    model_name = model or EMBEDDING_MODEL

    # -----------------------------------------------------
    # 1. LOKALE OLLAMA
    # -----------------------------------------------------

    embedding = get_ollama_embedding(
        text=clean_text,
        model=model_name,
    )

    if embedding is not None:
        return embedding

    # -----------------------------------------------------
    # 2. GEMINI FALLBACK
    # -----------------------------------------------------

    gemini_api_key = get_gemini_api_key()

    if gemini_api_key:
        embedding = get_gemini_embedding(
            text=clean_text,
            api_key=gemini_api_key,
        )

        if embedding is not None:
            return embedding

    # -----------------------------------------------------
    # 3. GEEN SILENTE NULVECTOR
    # -----------------------------------------------------

    logger.error(
        "Geen embedding-provider kon een embedding genereren. "
        "Ollama-model='%s'.",
        model_name,
    )

    return None


# ---------------------------------------------------------
# 8. NIET-GECACHETE VERSIE
# ---------------------------------------------------------

def get_embedding(
    text: str,
    model: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Niet-gecachete variant van get_embedding_cached().

    Handig voor situaties waarin direct een nieuwe embedding
    nodig is zonder gebruik te maken van Streamlit's cache.

    Args:
        text: Tekst waarvoor een embedding nodig is.
        model: Optioneel Ollama-model.

    Returns:
        numpy.ndarray of None.
    """

    clean_text = text.strip() if text else ""

    if not clean_text:
        return None

    model_name = model or EMBEDDING_MODEL

    # Ollama
    embedding = get_ollama_embedding(
        text=clean_text,
        model=model_name,
    )

    if embedding is not None:
        return embedding

    # Gemini
    gemini_api_key = get_gemini_api_key()

    if gemini_api_key:
        embedding = get_gemini_embedding(
            text=clean_text,
            api_key=gemini_api_key,
        )

        if embedding is not None:
            return embedding

    return None


# ---------------------------------------------------------
# 9. EMBEDDING DIMENSIE
# ---------------------------------------------------------

def get_embedding_dimension(
    embedding: Optional[np.ndarray],
) -> Optional[int]:
    """
    Geeft de daadwerkelijke dimensie van een embedding terug.

    Er wordt bewust geen globale EMBEDDING_DIMENSION gebruikt.

    Args:
        embedding: Een embedding-vector.

    Returns:
        Het aantal dimensies, of None.
    """

    if embedding is None:
        return None

    try:
        vector = np.asarray(embedding)

        if vector.ndim == 0:
            return None

        return int(vector.size)

    except Exception:
        return None


# ---------------------------------------------------------
# 10. MODEL SUPPORT CHECK
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def model_supports_embeddings(
    model_name: str,
) -> bool:
    """
    Controleert op basis van de modelnaam of een model
    waarschijnlijk embeddings ondersteunt.

    Dit is uitsluitend een snelle compatibiliteitscheck.
    De daadwerkelijke controle gebeurt pas wanneer Ollama
    het model aanspreekt.

    Args:
        model_name: Naam van het model.

    Returns:
        True wanneer de naam overeenkomt met bekende
        embedding-modellen.
    """

    if not model_name:
        return False

    normalized_name = model_name.lower().strip()

    known_keywords = (
        "embed",
        "embedding",
        "nomic",
        "mxbai",
        "arctic",
        "bge",
        "gte",
        "e5",
        "snowflake",
        "jina",
        "voyage",
        "all-minilm",
        "text-embedding",
        "multilingual-e5",
    )

    return any(
        keyword in normalized_name
        for keyword in known_keywords
    )


# ---------------------------------------------------------
# 11. EMBEDDING STATUS / DIAGNOSTIEK
# ---------------------------------------------------------

def get_embedding_status(
    model: Optional[str] = None,
) -> dict:
    """
    Geeft diagnostische informatie over de embedding-configuratie.

    Deze functie genereert zelf geen embedding.

    Returns:
        Dictionary met provider- en configuratiestatus.
    """

    model_name = model or EMBEDDING_MODEL
    gemini_key_available = bool(
        get_gemini_api_key()
    )

    ollama_online = ollama_available()

    return {
        "embedding_model": model_name,
        "ollama_available": ollama_online,
        "ollama_model_likely_supports_embeddings":
            model_supports_embeddings(model_name),
        "gemini_available": gemini_key_available,
        "gemini_embedding_model": GEMINI_EMBEDDING_MODEL,
    }


# ---------------------------------------------------------
# 12. CACHE RESET
# ---------------------------------------------------------

def clear_embedding_cache() -> None:
    """
    Leegt de Streamlit embedding-cache.

    Handig nadat het embedding-model in config.py is gewijzigd
    of nadat documenten opnieuw geïndexeerd moeten worden.
    """

    try:
        get_embedding_cached.clear()

    except Exception as exc:
        logger.warning(
            "Kon embedding-cache niet wissen: %s",
            exc,
        )
