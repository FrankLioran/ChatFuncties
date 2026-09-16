# ---- ai_router.py ----

import os
import time
import random
from dotenv import load_dotenv
from ollama import Client
from google import genai
from google.genai import types
from groq import Groq
from openai import OpenAI

from config import DEFAULT_TEMPERATURE
from safety import register_usage

from api_keys import (
    get_groq_api_key,
    get_gemini_api_key,
    get_openai_api_key,
)

# ---------------------------------------------------------
#  Gemini fallback
# ---------------------------------------------------------

GEMINI_PRIMARY_MODEL = "gemini-3.8-flash"
GEMINI_FALLBACK_MODEL = "gemini-3.6-flash"

# Alleen voor fouten waarbij opnieuw proberen zinvol is.
GEMINI_RETRY_DELAYS = [2, 6]

# ---------------------------------------------------------
#  Globale Ollama client + context
# ---------------------------------------------------------

OLLAMA_CLIENT = Client(host="http://127.0.0.1:11434")

# Basiscontext – wordt per model aangepast in get_ollama_options()
BASE_OLLAMA_CONTEXT = 8192

# KV-cache type (server-side via env, hier alleen voor awareness/debug)
OLLAMA_KV_CACHE = os.getenv("OLLAMA_KV_CACHE_TYPE", "q8_0")

load_dotenv()

# --- VEILIGE SESSION STATE FALLBACK ---
_cli_session_state = {}


def get_session_state():
    try:
        import streamlit as st
        from streamlit.runtime import exists
        if exists():
            return st.session_state
    except ImportError:
        pass
    return _cli_session_state


# ---------------------------------------------------------
#  KV-profiel per model (awareness)
# ---------------------------------------------------------

def get_kv_profile(model: str):
    """
    Geeft een logisch KV-profiel terug op basis van het model.
    """
    m = (model or "").lower()

    # Modellen die vaak beter draaien met zwaardere KV (fp16/bf16)
    if any(x in m for x in ["phi3:mini", "mistral:latest", "gemma3:4b", "gemma4:e2b"]):
        return "kv_heavy_pref"

    # Grote modellen → liever gequantized KV (q8_0 / q4_x)
    if any(x in m for x in ["8b", "9b", "gemma4:e4b", "gemma4:latest"]):
        return "kv_quant_pref"

    return "kv_neutral"


# ---------------------------------------------------------
#  Dynamische Ollama-opties per model + KV-cache awareness
# ---------------------------------------------------------

def get_ollama_options(model: str):
    ss = get_session_state()
    temperature = ss.get("temperature", DEFAULT_TEMPERATURE)

    m = (model or "").lower()
    kv = (OLLAMA_KV_CACHE or "").lower()
    kv_profile = get_kv_profile(model)

    num_ctx = BASE_OLLAMA_CONTEXT
    num_predict = 1024
    num_gpu = 999
    num_batch = 32

    # SPECIALE PROFIELEN VOOR GEMMA 4
    if "gemma4:latest" in m or (m.startswith("gemma4") and "latest" in m):
        num_ctx = 8192
        num_predict = 4096
        num_batch = 64

        ss["debug_ollama_options"] = {
            "model": model,
            "kv_cache_type": kv,
            "kv_profile": kv_profile,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "num_gpu": num_gpu,
            "num_batch": num_batch,
            "special_profile": "gemma4:latest",
        }

        return {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_gpu": num_gpu,
            "num_batch": num_batch,
            "num_thread": 0,
            "num_ctx": num_ctx,
        }

    if "gemma4:e2b" in m:
        num_ctx = 6144
        num_predict = 2048
        num_batch = 48

        ss["debug_ollama_options"] = {
            "model": model,
            "kv_cache_type": kv,
            "kv_profile": kv_profile,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "num_gpu": num_gpu,
            "num_batch": num_batch,
            "special_profile": "gemma4:e2b",
        }

        return {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_gpu": num_gpu,
            "num_batch": num_batch,
            "num_thread": 0,
            "num_ctx": num_ctx,
        }

    if "gemma4:e4b" in m:
        num_ctx = 4096
        num_predict = 1536
        num_batch = 32

        ss["debug_ollama_options"] = {
            "model": model,
            "kv_cache_type": kv,
            "kv_profile": kv_profile,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "num_gpu": num_gpu,
            "num_batch": num_batch,
            "special_profile": "gemma4:e4b",
        }

        return {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_gpu": num_gpu,
            "num_batch": num_batch,
            "num_thread": 0,
            "num_ctx": num_ctx,
        }

    # NORMALE PROFIELEN
    if "3b" in m or "1b" in m or "mini" in m:
        num_ctx = 16384
        num_predict = 4096
        num_batch = 64
    elif "4b" in m:
        num_ctx = 12288
        num_predict = 3072
        num_batch = 64
    elif "7b" in m:
        num_ctx = 8192
        num_predict = 2048
        num_batch = 48
    elif "8b" in m:
        num_ctx = 6144
        num_predict = 1536
        num_batch = 48
    elif "9b" in m:
        num_ctx = 4096
        num_predict = 1024
        num_batch = 32

    if kv in {"f16", "bf16"}:
        if "7b" in m or "8b" in m or "9b" in m or "e4b" in m:
            num_ctx = min(num_ctx, 4096)
    elif kv in {"q4_0", "q4_1", "iq4_nl"}:
        num_ctx = int(num_ctx * 1.25)

    ss["debug_ollama_options"] = {
        "model": model,
        "kv_cache_type": kv,
        "kv_profile": kv_profile,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "num_gpu": num_gpu,
        "num_batch": num_batch,
    }

    return {
        "temperature": temperature,
        "num_predict": num_predict,
        "num_gpu": num_gpu,
        "num_batch": num_batch,
        "num_thread": 0,
        "num_ctx": num_ctx,
    }


# ---------------------------------------------------------
#  Streaming & Default Modellen
# ---------------------------------------------------------

def stream_ollama(messages, model):
    options = get_ollama_options(model)

    for chunk in OLLAMA_CLIENT.chat(
        model=model,
        messages=messages,
        stream=True,
        options=options,
    ):
        token = chunk.message.content or ""
        if token:
            yield token

DEFAULT_MODELS = {
    "Lokaal": "qwen2.5:7b",
    "Groq": "openai/gpt-oss-120b",
    "Gemini": GEMINI_PRIMARY_MODEL,
    "OpenAI": "gpt-5.6-luna",
}

def debug_provider(provider, model, messages):
    print("=" * 50)
    print(f"Provider : {provider}")
    print(f"Model    : {model}")

    if provider.startswith("Lokaal"):
        print(f"KV-cache : {OLLAMA_KV_CACHE}")
        print(f"KV-prof  : {get_kv_profile(model)}")

    print("=" * 50)

def get_active_model(provider):
    ss = get_session_state()
    model = ss.get("model_name")
    if not model:
        model = DEFAULT_MODELS.get(provider, "qwen2.5:3b")
    return model

def ask_openai(messages, model=None):
    ss = get_session_state()
    api_key = get_openai_api_key()

    if not api_key:
        return "[Geen OpenAI API key gevonden.]"

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model or "gpt-5.6-luna",
        input=messages,
    )

    total_tokens = 0
    try:
        total_tokens = response.usage.total_tokens
    except Exception:
        pass

    register_usage(total_tokens)

    return response.output_text.strip()

# ---------------------------------------------------------
#  Router Functions
# ---------------------------------------------------------

def ask_ai(messages):
    ss = get_session_state()
    provider = ss.get("ai_provider", "Lokaal")
    model = get_active_model(provider)

    ss["debug_active_model"] = f"{provider} - {model}"

    if provider == "Gemini":
        return ask_gemini(messages, model)

    if provider == "Groq":
        return ask_groq(messages, model)

    if provider == "OpenAI":
        return ask_openai(messages, model)

    return ask_ollama(messages, model)

def ask_ollama(messages, model=None, provider_name=None):
    ss = get_session_state()

    if not model:
        model = get_active_model("Lokaal")

    debug_provider(
        provider_name or ss.get("ai_provider", "Lokaal"),
        model,
        messages,
    )

    options = get_ollama_options(model)

    reply = OLLAMA_CLIENT.chat(
        model=model,
        messages=messages,
        options=options,
    )

    register_usage(0)
    return reply.message.content.strip()

def _is_retryable_gemini_error(error):
    """
    Bepaal of een Gemini-fout tijdelijk kan zijn
    en opnieuw proberen zinvol is.
    """

    # Eerst proberen een expliciete statuscode te vinden.
    status = getattr(error, "status_code", None)

    if status is None:
        status = getattr(error, "code", None)

    if status in {429, 500, 502, 503, 504}:
        return True

    # Fallback voor SDK-exceptions waarbij alleen tekst beschikbaar is.
    text = str(error).lower()

    retryable_markers = [
        "503",
        "unavailable",
        "service unavailable",
        "500",
        "internal server error",
        "502",
        "bad gateway",
        "504",
        "deadline exceeded",
        "429",
        "resource exhausted",
        "rate limit",
        "too many requests",
    ]

    return any(marker in text for marker in retryable_markers)


def _build_gemini_prompt(messages):
    """
    Zet onze interne message-structuur om naar één prompt.
    """
    parts = []

    for msg in messages:
        role = msg["role"]

        if role == "system":
            parts.append(f"SYSTEEM:\n{msg['content']}")
        elif role == "user":
            parts.append(f"GEBRUIKER:\n{msg['content']}")
        else:
            parts.append(f"ASSISTENT:\n{msg['content']}")

    return "\n\n".join(parts)


def _call_gemini_with_retry(client, model, prompt, temperature):
    """
    Eén Gemini-model aanroepen met beperkte retry/backoff.

    Belangrijk:
    - 503/429/5xx → beperkte retry
    - andere fouten → direct doorgeven
    """

    last_error = None

    for attempt, base_delay in enumerate(GEMINI_RETRY_DELAYS, start=1):

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature
                ),
            )

            return response

        except Exception as e:
            last_error = e

            if not _is_retryable_gemini_error(e):
                raise

            if attempt >= len(GEMINI_RETRY_DELAYS):
                break

            # Kleine jitter voorkomt synchroon retry-gedrag.
            delay = base_delay + random.uniform(0, 1)

            print(
                f"[Eva] Gemini '{model}' tijdelijk niet beschikbaar "
                f"(poging {attempt}/{len(GEMINI_RETRY_DELAYS)}). "
                f"Nieuwe poging over {delay:.1f}s."
            )

            time.sleep(delay)

    raise last_error


def ask_gemini(messages, model=None):
    ss = get_session_state()

    api_key = get_gemini_api_key()

    if not api_key:
        return "[Geen Gemini API key gevonden]"

    client = genai.Client(api_key=api_key)

    primary_model = model or GEMINI_PRIMARY_MODEL
    prompt = _build_gemini_prompt(messages)

    # Alleen gebruiken als de primaire daadwerkelijk faalt
    # door een permanente/model-specifieke fout.
    fallback_model = (
        GEMINI_FALLBACK_MODEL
        if primary_model != GEMINI_FALLBACK_MODEL
        else None
    )

    last_error = None

    # -----------------------------------------------------
    # Primaire model
    # -----------------------------------------------------

    debug_provider("Gemini", primary_model, messages)

    try:
        response = _call_gemini_with_retry(
            client=client,
            model=primary_model,
            prompt=prompt,
            temperature=ss.get(
                "temperature",
                DEFAULT_TEMPERATURE
            ),
        )

        try:
            total_tokens = (
                response.usage_metadata.total_token_count
            )
        except Exception:
            total_tokens = 0

        register_usage(total_tokens)

        return response.text.strip()

    except Exception as e:
        last_error = e

        print(
            f"[Eva] Primair Gemini-model "
            f"'{primary_model}' faalde: {e}"
        )

        # -------------------------------------------------
        # Alleen fallback bij NIET-transiënte fout
        # -------------------------------------------------

        if _is_retryable_gemini_error(e):
            return (
                "[Gemini is tijdelijk niet beschikbaar. "
                "Eva heeft het opnieuw geprobeerd, maar "
                "de dienst bleef onbeschikbaar.]"
            )

    # -----------------------------------------------------
    # Eén gecontroleerde fallback
    # -----------------------------------------------------

    if fallback_model:
        debug_provider("Gemini-Fallback", fallback_model, messages)

        try:
            response = _call_gemini_with_retry(
                client=client,
                model=fallback_model,
                prompt=prompt,
                temperature=ss.get(
                    "temperature",
                    DEFAULT_TEMPERATURE
                ),
            )

            try:
                total_tokens = (
                    response.usage_metadata.total_token_count
                )
            except Exception:
                total_tokens = 0

            register_usage(total_tokens)

            return response.text.strip()

        except Exception as e:
            last_error = e

            print(
                f"[Eva] Gemini fallback-model "
                f"'{fallback_model}' faalde: {e}"
            )

    return (
        f"[Geen Gemini-model beschikbaar. "
        f"Laatste fout: {last_error}]"
    )

GROQ_FALLBACK_ORDER = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]


def ask_groq(messages, model=None):
    ss = get_session_state()
    api_key = get_groq_api_key()
    if not api_key:
        return "[Geen Groq API key gevonden.]"

    client = Groq(api_key=api_key)
    try_order = [model] + [m for m in GROQ_FALLBACK_ORDER if m != model] if model else GROQ_FALLBACK_ORDER

    last_error = None
    for candidate in try_order:
        debug_provider("Groq", candidate, messages)
        try:
            response = client.chat.completions.create(
                model=candidate,
                messages=messages,
                temperature=ss.get("temperature", DEFAULT_TEMPERATURE),
            )

            try:
                total_tokens = response.usage.total_tokens
            except Exception:
                total_tokens = 0

            register_usage(total_tokens)
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[Eva] Model '{candidate}' faalde ({e})")
            last_error = e
            continue

    return f"[Geen Groq-model beschikbaar. Laatste fout: {last_error}]"


def stream_ai(messages):
    ss = get_session_state()
    provider = ss.get("ai_provider", "Lokaal")
    model = get_active_model(provider)

    if provider == "Lokaal":
        return stream_ollama(messages, model)

    return None


# ---------------------------------------------------------
#  Benchmark / Stress-test Helpers
# ---------------------------------------------------------

def benchmark_model(model: str, prompt: str = "Testprompt voor benchmark."):
    messages = [{"role": "user", "content": prompt}]
    debug_provider("Lokaal-Benchmark", model, messages)

    options = get_ollama_options(model)

    start = time.time()
    reply = OLLAMA_CLIENT.chat(
        model=model,
        messages=messages,
        options=options,
    )
    end = time.time()

    text = reply.message.content
    duration = end - start
    tokens = max(len(text.split()), 1)
    tps = tokens / duration

    return {
        "model": model,
        "duration": duration,
        "tokens": tokens,
        "tokens_per_sec": tps,
    }


def stress_test_context(model: str, base_prompt: str = "Contextblok"):
    long_prompt = "\n".join([base_prompt] * 64)
    return benchmark_model(model, long_prompt)


def stress_test_streaming(model: str, prompt: str = "Streaming test."):
    messages = [{"role": "user", "content": prompt}]
    debug_provider("Lokaal-Streaming-Test", model, messages)

    options = get_ollama_options(model)

    start = time.time()
    total_tokens = 0

    for chunk in OLLAMA_CLIENT.chat(
        model=model,
        messages=messages,
        stream=True,
        options=options,
    ):
        # Aangepast naar object-attribuut voor de nieuwere Ollama SDK
        token = chunk.message.content or ""
        if token:
            total_tokens += len(token.split())

    end = time.time()
    duration = end - start
    tps = total_tokens / duration if duration > 0 else 0.0

    return {
        "model": model,
        "duration": duration,
        "tokens": total_tokens,
        "tokens_per_sec": tps,
    }