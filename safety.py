# safety.py
import streamlit as st

def _get_safe_state():
    try:
        return st.session_state
    except Exception:
        return None

def check_limits():
    ss = _get_safe_state()
    if not ss or not getattr(ss, "safe_mode", False):
        return

    if getattr(ss, "requests_used", 0) >= getattr(ss, "request_limit", 100):
        raise RuntimeError("Maximum aantal requests bereikt.")

    if getattr(ss, "tokens_used", 0) >= getattr(ss, "token_limit", 1000000):
        raise RuntimeError("Maximum aantal tokens bereikt.")

def register_usage(total_tokens):
    ss = _get_safe_state()
    if not ss:
        return  # Buiten Streamlit (bijv. benchmark) hoeven we niets bij te houden

    try:
        provider = ss.get("ai_provider")
        if "requests_used" in ss:
            ss.requests_used += 1
        if provider != "Lokaal" and "tokens_used" in ss:
            ss.tokens_used += total_tokens
    except Exception:
        pass