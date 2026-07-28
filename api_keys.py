import json
import logging
from pathlib import Path

import streamlit as st

LOCAL_SECRETS_PATH = Path("secrets.json")


def _get_api_key(key_name: str):
    """Haalt een API-key op uit session_state, Streamlit Secrets of secrets.json."""

    # 1. Handmatig ingevoerd door gebruiker
    session_key = st.session_state.get(f"{key_name}_user", "")
    if session_key:
        return session_key.strip()

    # 2. Streamlit Community Cloud Secrets
    try:
        value = st.secrets.get(key_name)
        if value:
            return value.strip()
    except Exception:
        pass

    # 3. Lokaal secrets.json
    if LOCAL_SECRETS_PATH.exists():
        try:
            with open(LOCAL_SECRETS_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            if isinstance(loaded, dict):
                value = loaded.get(key_name)
                if value:
                    return value.strip()

        except Exception as exc:
            logging.debug(f"Kon {LOCAL_SECRETS_PATH} niet lezen: {exc}")

    return None


def get_gemini_api_key():
    return _get_api_key("gemini_api_key")


def get_groq_api_key():
    return _get_api_key("groq_api_key")