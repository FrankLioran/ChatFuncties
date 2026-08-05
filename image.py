# image.py
import os
import time
import requests
import urllib.parse
import streamlit as st
from pathlib import Path
from config import OUTPUT_FOLDER_PAINT

def generate_and_save_image(prompt_text: str, retries: int = 2) -> Path | None:
    """
    Genereert een afbeelding via Pollinations AI.
    Volledig gratis, cloud-based (0% VRAM) met automatische herhaalpogingen.
    """
    encoded_prompt = urllib.parse.quote(prompt_text)
    enhanced_url = f"https://image.pollinations.ai/p/{encoded_prompt}?width=1024&height=1024&nologo=true&enhance=true"

    # Bepaal de uitvoermap (gebruikt OUTPUT_FOLDER_PAINT of valt terug op een lokale map)
    if OUTPUT_FOLDER_PAINT:
        output_dir = Path(OUTPUT_FOLDER_PAINT)
    else:
        BASE_DIR = Path(__file__).resolve().parent
        output_dir = BASE_DIR / "Generated_images"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Probeer het 'retries' aantal keren als de cloud even niet reageert
    for attempt in range(retries + 1):
        try:
            response = requests.get(enhanced_url, timeout=60)

            if response.status_code == 200 and len(response.content) > 0:
                # 1. Bepaal EERST de bestandsnaam en het pad
                file_name = f"generated_{os.urandom(4).hex()}.png"
                image_path = output_dir / file_name

                # 2. Sla de bytes op in het bestand
                image_path.write_bytes(response.content)

                # 3. Geef het pad terug naar app.py
                return image_path
            else:
                print(f"Pollinations statuscode {response.status_code} op poging {attempt + 1}")

        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            print(f"Fout bij poging {attempt + 1}: {e}")

        # Wacht 2 seconden voor we het opnieuw proberen
        if attempt < retries:
            time.sleep(2)

    # Als alle pogingen zijn mislukt:
    st.error("Het genereren van de afbeelding via Pollinations is helaas mislukt na meerdere pogingen.")
    return None
