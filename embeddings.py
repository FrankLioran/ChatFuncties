# embeddings.py
# ---------------------------------------------------------
# ROBUUSTE EMBEDDING ENGINE
#
# Primaire provider : lokale Ollama
# Fallback provider : Gemini
#
# Eigenschappen:
# - single + batch embeddings
# - Ollama response-compatibiliteit
# - Gemini 429/5xx retry met backoff
# - batch-output behoudt altijd inputlengte en volgorde
# - geen stille nulvectoren
# - Streamlit cache
# - diagnostiek
# ---------------------------------------------------------

import logging
import os
import random
import time
from typing import List, Optional

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

# Dit zijn wachttijden tussen nieuwe pogingen.
# [2, 5, 10] betekent maximaal drie pogingen totaal.
GEMINI_RETRY_DELAYS = [2, 5, 10]

# Kleine bescherming tegen directe opeenvolgende REST-calls.
GEMINI_MIN_INTERVAL_SEC = 0.05

_last_gemini_call_time = 0.0


# ---------------------------------------------------------
# 2. OLLAMA BESCHIKBAARHEID
# ---------------------------------------------------------

@st.cache_resource(ttl=300)
def ollama_available() -> bool:
    """
    Controleert uitsluitend of Ollama bereikbaar is.

    Er wordt hier bewust niet gecontroleerd of een specifiek
    embedding-model aanwezig is.
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
        3. environment variable
    """

    # 1. Session state
    try:
        session_key = st.session_state.get(
            "gemini_api_key_user"
        )
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
    Converteert embedding-data naar een 1D float32 numpy-array.

    Er wordt bewust geen vaste dimensie afgedwongen.
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
            "%s vectorconversie mislukt: %s",
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
            "%s gaf NaN/Inf-waarden terug.",
            provider,
        )
        return None

    return vector


# ---------------------------------------------------------
# 5. OLLAMA SINGLE EMBEDDING
# ---------------------------------------------------------

def get_ollama_embedding(
    text: str,
    model: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Genereert één embedding via Ollama.
    """

    clean_text = text.strip() if text else ""

    if not clean_text:
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

    embedding_data = None

    # Moderne Ollama response:
    # {"embeddings": [[...]]}
    if isinstance(response, dict):
        embeddings = response.get("embeddings")

        if isinstance(embeddings, list) and embeddings:
            embedding_data = embeddings[0]

        # Oudere response:
        # {"embedding": [...]}
        elif isinstance(response.get("embedding"), list):
            embedding_data = response.get("embedding")

    else:
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

    if vector is not None:
        logger.debug(
            "Ollama embedding succesvol: model=%s, dimensie=%d",
            model_name,
            vector.shape[0],
        )

    return vector


# ---------------------------------------------------------
# 6. OLLAMA BATCH EMBEDDINGS
# ---------------------------------------------------------

def get_ollama_embeddings_batch(
    texts: List[str],
    model: Optional[str] = None,
) -> List[Optional[np.ndarray]]:
    """
    Genereert embeddings voor meerdere teksten.

    Belangrijk:
        De lengte en volgorde van de output zijn altijd gelijk
        aan die van de input.

        Lege invoer -> None op dezelfde positie.
    """

    if not texts:
        return []

    model_name = model or EMBEDDING_MODEL

    # Output vooraf op exacte inputlengte.
    results: List[Optional[np.ndarray]] = [None] * len(texts)

    # Alleen niet-lege teksten naar Ollama sturen.
    valid_items = [
        (index, text.strip())
        for index, text in enumerate(texts)
        if text and text.strip()
    ]

    if not valid_items:
        return results

    if not ollama_available():
        return results

    clean_texts = [text for _, text in valid_items]

    try:
        response = ollama.embed(
            model=model_name,
            input=clean_texts,
        )
    except Exception as exc:
        logger.warning(
            "Ollama batch embedding mislukt voor model '%s': %s",
            model_name,
            exc,
        )
        return results

    raw_embeddings = []

    if isinstance(response, dict):
        raw_embeddings = response.get(
            "embeddings",
            [],
        )
    else:
        raw_embeddings = getattr(
            response,
            "embeddings",
            [],
        )

    if not raw_embeddings:
        logger.warning(
            "Ollama gaf geen batch embeddings terug voor '%s'.",
            model_name,
        )
        return results

    # Normaal gesproken evenveel vectors als geldige inputs.
    # We beschermen ons tegen een afwijkende response.
    count = min(
        len(valid_items),
        len(raw_embeddings),
    )

    for pos in range(count):
        original_index = valid_items[pos][0]

        vector = _validate_embedding(
            raw_embeddings[pos],
            provider=f"Ollama/{model_name}",
        )

        results[original_index] = vector

    if len(raw_embeddings) != len(valid_items):
        logger.warning(
            "Ollama batchlengte onverwacht: "
            "inputs=%d, embeddings=%d.",
            len(valid_items),
            len(raw_embeddings),
        )

    return results


# ---------------------------------------------------------
# 7. GEMINI PACING
# ---------------------------------------------------------

def _pacing_gemini() -> None:
    """
    Zorgt dat opeenvolgende Gemini REST-calls niet exact
    gelijktijdig plaatsvinden.
    """

    global _last_gemini_call_time

    now = time.time()
    delta = now - _last_gemini_call_time

    if delta < GEMINI_MIN_INTERVAL_SEC:
        time.sleep(
            GEMINI_MIN_INTERVAL_SEC - delta
        )

    _last_gemini_call_time = time.time()


# ---------------------------------------------------------
# 8. GEMINI RETRY DETECTIE
# ---------------------------------------------------------

def _gemini_retryable_status(status_code: int) -> bool:
    """
    Alleen tijdelijke server/rate-limit fouten worden opnieuw
    geprobeerd.

    403 wordt bewust NIET opnieuw geprobeerd.
    """
    return status_code in {
        429,  # rate limit
        500,  # internal server error
        502,  # bad gateway
        503,  # unavailable
        504,  # gateway timeout
    }


# ---------------------------------------------------------
# 9. GEMINI SINGLE EMBEDDING
# ---------------------------------------------------------

def get_gemini_embedding(
    text: str,
    api_key: str,
    model: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Genereert één Gemini embedding.

    De single-call variant gebruikt dezelfde retrystrategie
    als de batchvariant.
    """

    clean_text = text.strip() if text else ""

    if not clean_text or not api_key:
        return None

    results = get_gemini_embeddings_batch(
        [clean_text],
        api_key=api_key,
        model=model,
    )

    return results[0] if results else None


# ---------------------------------------------------------
# 10. GEMINI BATCH EMBEDDINGS
# ---------------------------------------------------------

def get_gemini_embeddings_batch(
    texts: List[str],
    api_key: str,
    model: Optional[str] = None,
) -> List[Optional[np.ndarray]]:
    """
    Genereert Gemini embeddings via batchEmbedContents.

    De lengte en volgorde van de output zijn gelijk aan de input.

    429 en tijdelijke 5xx-fouten worden opnieuw geprobeerd.
    403 wordt NIET opnieuw geprobeerd.
    """

    if not texts:
        return []

    results: List[Optional[np.ndarray]] = [None] * len(texts)

    if not api_key:
        return results

    model_name = model or GEMINI_EMBEDDING_MODEL

    valid_items = [
        (index, text.strip())
        for index, text in enumerate(texts)
        if text and text.strip()
    ]

    if not valid_items:
        return results

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model_name}:batchEmbedContents"
    )

    headers = {
        "Content-Type": "application/json",
    }

    requests_payload = [
        {
            "model": f"models/{model_name}",
            "content": {
                "parts": [
                    {
                        "text": text,
                    }
                ]
            },
        }
        for _, text in valid_items
    ]

    payload = {
        "requests": requests_payload
    }

    # Eerste poging direct, daarna de opgegeven wachttijden.
    retry_delays = [0] + GEMINI_RETRY_DELAYS

    for attempt, delay in enumerate(
        retry_delays,
        start=1,
    ):
        if delay > 0:
            jitter = random.uniform(
                0.1,
                0.5,
            )

            wait_time = delay + jitter

            logger.info(
                "Gemini embedding tijdelijk niet beschikbaar. "
                "Nieuwe poging over %.2f sec.",
                wait_time,
            )

            time.sleep(wait_time)

        _pacing_gemini()

        try:
            response = requests.post(
                url,
                headers=headers,
                params={"key": api_key},
                json=payload,
                timeout=30,
            )

        except requests.exceptions.Timeout:
            logger.warning(
                "Gemini embedding timeout "
                "(poging %d/%d).",
                attempt,
                len(retry_delays),
            )

            if attempt >= len(retry_delays):
                break

            continue

        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Netwerkfout Gemini embedding "
                "(poging %d/%d): %s",
                attempt,
                len(retry_delays),
                exc,
            )

            if attempt >= len(retry_delays):
                break

            continue

        status = response.status_code

        if _gemini_retryable_status(status):
            logger.warning(
                "Gemini embedding HTTP %d "
                "(poging %d/%d).",
                status,
                attempt,
                len(retry_delays),
            )

            if attempt >= len(retry_delays):
                break

            continue

        if status == 403:
            # Bewust geen retry-loop.
            logger.error(
                "Gemini embedding gaf HTTP 403 Forbidden. "
                "De aanvraag wordt niet opnieuw geprobeerd."
            )
            return results

        if status < 200 or status >= 300:
            logger.error(
                "Gemini embedding HTTP %d: %s",
                status,
                response.text[:500],
            )
            return results

        try:
            data = response.json()
        except ValueError as exc:
            logger.error(
                "Gemini gaf geen geldige JSON-response: %s",
                exc,
            )
            return results

        raw_embeddings = data.get(
            "embeddings",
            [],
        )

        if not isinstance(
            raw_embeddings,
            list,
        ):
            logger.error(
                "Gemini batchresponse bevat geen geldige "
                "'embeddings'-lijst."
            )
            return results

        count = min(
            len(valid_items),
            len(raw_embeddings),
        )

        for pos in range(count):
            original_index = valid_items[pos][0]

            item = raw_embeddings[pos]

            values = (
                item.get("values", [])
                if isinstance(item, dict)
                else []
            )

            vector = _validate_embedding(
                values,
                provider=f"Gemini/{model_name}",
            )

            results[original_index] = vector

        if len(raw_embeddings) != len(valid_items):
            logger.warning(
                "Gemini batchlengte onverwacht: "
                "inputs=%d, embeddings=%d.",
                len(valid_items),
                len(raw_embeddings),
            )

        return results

    logger.error(
        "Gemini batch embedding finaal mislukt "
        "na %d pogingen.",
        len(retry_delays),
    )

    return results


# ---------------------------------------------------------
# 11. CENTRALE BATCH ROUTING
# ---------------------------------------------------------

def get_embeddings_batch(
    texts: List[str],
    model: Optional[str] = None,
) -> List[Optional[np.ndarray]]:
    """
    Centrale batch embeddingfunctie.

    Strategie:
        1. probeer de volledige batch lokaal via Ollama
        2. als de lokale batch niet volledig lukt:
           probeer de volledige batch via Gemini
        3. nooit stilletjes Ollama- en Gemini-vectoren
           door elkaar mengen binnen dezelfde batch

    De output heeft altijd dezelfde lengte als texts.
    """

    if not texts:
        return []

    model_name = model or EMBEDDING_MODEL

    # -----------------------------------------------------
    # 1. OLLAMA
    # -----------------------------------------------------

    if ollama_available():
        ollama_results = get_ollama_embeddings_batch(
            texts,
            model=model_name,
        )

        if (
            len(ollama_results) == len(texts)
            and all(
                vector is not None
                for vector in ollama_results
            )
        ):
            return ollama_results

        logger.warning(
            "Ollama kon niet alle embeddings van de batch "
            "genereren; Gemini wordt voor de volledige batch "
            "als fallback geprobeerd."
        )

    # -----------------------------------------------------
    # 2. GEMINI
    # -----------------------------------------------------

    api_key = get_gemini_api_key()

    if api_key:
        gemini_results = get_gemini_embeddings_batch(
            texts,
            api_key=api_key,
        )

        return gemini_results

    # -----------------------------------------------------
    # 3. GEEN PROVIDER
    # -----------------------------------------------------

    return [None] * len(texts)


# ---------------------------------------------------------
# 12. GECACHETE SINGLE EMBEDDING
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_embedding_cached(
    text: str,
    model: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Centrale gecachete single embedding.

    Provider-volgorde:
        Ollama -> Gemini
    """

    clean_text = text.strip() if text else ""

    if not clean_text:
        return None

    model_name = model or EMBEDDING_MODEL

    embedding = get_ollama_embedding(
        text=clean_text,
        model=model_name,
    )

    if embedding is not None:
        return embedding

    gemini_api_key = get_gemini_api_key()

    if gemini_api_key:
        embedding = get_gemini_embedding(
            text=clean_text,
            api_key=gemini_api_key,
        )

        if embedding is not None:
            return embedding

    logger.error(
        "Geen embedding-provider kon '%s' embedden.",
        model_name,
    )

    return None


# ---------------------------------------------------------
# 13. NIET-GECACHETE SINGLE EMBEDDING
# ---------------------------------------------------------

def get_embedding(
    text: str,
    model: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Niet-gecachete variant van get_embedding_cached().
    """

    clean_text = text.strip() if text else ""

    if not clean_text:
        return None

    model_name = model or EMBEDDING_MODEL

    embedding = get_ollama_embedding(
        text=clean_text,
        model=model_name,
    )

    if embedding is not None:
        return embedding

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
# 14. EMBEDDING DIMENSIE
# ---------------------------------------------------------

def get_embedding_dimension(
    embedding: Optional[np.ndarray],
) -> Optional[int]:
    """
    Geeft de daadwerkelijke dimensie van een embedding terug.
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
# 15. MODEL SUPPORT CHECK
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def model_supports_embeddings(
    model_name: str,
) -> bool:
    """
    Snelle naamgebaseerde indicatie of een model waarschijnlijk
    een embedding-model is.

    Dit vervangt geen daadwerkelijke Ollama-test.
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
# 16. EMBEDDING STATUS
# ---------------------------------------------------------

def get_embedding_status(
    model: Optional[str] = None,
) -> dict:
    """
    Geeft diagnostische informatie zonder een embedding
    te genereren.
    """

    model_name = model or EMBEDDING_MODEL

    return {
        "embedding_model": model_name,
        "ollama_available": ollama_available(),
        "ollama_model_likely_supports_embeddings":
            model_supports_embeddings(model_name),
        "gemini_available": bool(
            get_gemini_api_key()
        ),
        "gemini_embedding_model":
            GEMINI_EMBEDDING_MODEL,
    }


# ---------------------------------------------------------
# 17. CACHE RESET
# ---------------------------------------------------------

def clear_embedding_cache() -> None:
    """
    Leegt de Streamlit embedding-cache.
    """

    try:
        get_embedding_cached.clear()
    except Exception as exc:
        logger.warning(
            "Kon embedding-cache niet wissen: %s",
            exc,
        )
