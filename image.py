# image.py
import os
import requests
import streamlit as st
from pathlib import Path
import urllib.parse

def generate_and_save_image(prompt_text: str) -> Path | None:
    """
    Genereert een prachtige afbeelding via Pollinations AI.
    Volledig gratis, cloud-based (0% VRAM) Dit voorkomt overbelasting van de GTX 1050.
    """
    try:
        # Maak de tekst veilig voor gebruik in een URL
        encoded_prompt = urllib.parse.quote(prompt_text)
        
        # FIX: De exacte en juiste URL-structuur met /p/ erin
        enhanced_url = f"https://image.pollinations.ai/p/{encoded_prompt}?width=1024&height=1024&nologo=true&enhance=true"
        
        # Haal de afbeelding op uit de cloud
        response = requests.get(enhanced_url, timeout=30)
        
        if response.status_code == 200:
            # Map controleren / aanmaken op Ubuntu

            BASE_DIR = Path(__file__).resolve().parent
            output_dir = BASE_DIR / "Generated_images"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Sla de afbeelding op met een unieke naam
            image_path = output_dir / f"generated_{os.urandom(4).hex()}.png"
            image_path.write_bytes(response.content)
            return image_path
        else:
            st.error(f"Pollinations AI gaf statuscode: {response.status_code}")
            return None

    except Exception as e:
        st.error(f"Fout bij genereren afbeelding via cloud: {e}")
        return None
