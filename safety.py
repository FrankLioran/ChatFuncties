import streamlit as st


def check_limits():

    if not st.session_state.safe_mode:
        return

    if st.session_state.requests_used >= st.session_state.request_limit:

        raise RuntimeError(
            "Maximum aantal requests bereikt."
        )

    if st.session_state.tokens_used >= st.session_state.token_limit:

        raise RuntimeError(
            "Maximum aantal tokens bereikt."
        )


def register_usage(total_tokens):

    provider = st.session_state.get("ai_provider")
    st.session_state.requests_used += 1
    if provider != "Lokaal":
        st.session_state.tokens_used += total_tokens
    if provider == "Lokaal":
        return