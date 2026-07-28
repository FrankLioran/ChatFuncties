# audio.py — Hugging Face compatibele versie
import streamlit as st
from google import genai
from api_keys import get_gemini_api_key

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
    api_key = get_gemini_api_key()
    
    if not api_key:
        st.warning("Geen Gemini API key gevonden.")
        return None
    
    client = genai.Client(api_key=api_key)    

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
