# audio.py — Hugging Face compatibele versie
import streamlit as st
import google.generativeai as genai
from config import GEMINI_API_KEY

def listen_and_transcribe():
    """
    Hugging Face heeft geen microfoonondersteuning.
    Deze functie toont een melding en biedt optioneel
    audio-bestand upload + transcriptie via Gemini.
    """

    st.warning("Live microfoon-invoer is niet beschikbaar op Hugging Face Spaces.")

    # Optioneel: audio-bestand upload
    uploaded_audio = st.file_uploader(
        "Upload een audiobestand (WAV/MP3) om te transcriberen",
        type=["wav", "mp3"]
    )

    if not uploaded_audio:
        return None

    # Configureer Gemini
    if not GEMINI_API_KEY:
        st.error("Geen GEMINI_API_KEY gevonden.")
        return None

    genai.configure(api_key=GEMINI_API_KEY)

    try:
        audio_bytes = uploaded_audio.read()

        model = genai.GenerativeModel("gemini-2.5-flash-lite")

        response = model.generate_content(
            [
                {
                    "mime_type": "audio/wav",
                    "data": audio_bytes
                }
            ],
            generation_config={"temperature": 0.0}
        )

        transcript = response.text.strip()
        st.success(f"Transcriptie: {transcript}")
        return transcript

    except Exception as e:
        st.error(f"Fout bij transcriberen: {e}")
        return None
