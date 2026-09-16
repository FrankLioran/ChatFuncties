# audio.py — Hugging Face compatibele versie
import streamlit as st
from google import genai
from google.genai import types
from api_keys import get_gemini_api_key

def listen_and_transcribe():
    """
    Biedt de mogelijkheid om een audiobestand (WAV, MP3, M4A, OGG)
    te uploaden en te transcriberen via Gemini (Nieuwe Google GenAI SDK).
    """

    st.warning("Live microfoon-invoer is niet beschikbaar op cloud-omgevingen.")

    # 1. Bestand uploaden
    uploaded_audio = st.file_uploader(
        "Upload een audiobestand om te transcriberen",
        type=["wav", "mp3", "m4a", "ogg"]
    )

    if not uploaded_audio:
        return None

    # 2. API-sleutel controleren
    api_key = get_gemini_api_key()
    if not api_key:
        st.warning("Geen Gemini API key gevonden.")
        return None

    client = genai.Client(api_key=api_key)

    try:
        audio_bytes = uploaded_audio.read()

        # Bepaal het mime-type netjes dynamisch
        mime_type = uploaded_audio.type or "audio/wav"

        # 3. Transcriptie aanvragen via de nieuwe SDK
        with st.spinner("Audio transcriberen via Gemini..."):
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[
                    "Transcribeer de volgende audio exact in de gesproken taal:",
                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type=mime_type,
                    ),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0
                )
            )

        transcript = response.text.strip() if response.text else ""

        if transcript:
            st.success(f"**Transcriptie:** {transcript}")
            return transcript
        else:
            st.warning("Geen spraak kunnen detecteren in het bestand.")
            return None

    except Exception as e:
        st.error(f"Fout bij transcriberen: {e}")
        return None