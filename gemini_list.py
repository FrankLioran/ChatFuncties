# Filnaam: gemini_model_lister.py
import os
from google import genai
from api_keys import get_gemini_api_key

def list_available_models():
    """
    Haalt een lijst van beschikbare modellen op basis van de ingelogde API-sleutel.
    """
    try:
        # Zorg ervoor dat je API-sleutel ingesteld is in je environment variables
        client = genai.Client()

        print("\n==================================================")
        print("🤖 Beschikbare Gemini Modellen (via API Client) 🤖")
        print("==================================================")

        # De client heeft een methode om de lijst van beschikbare modellen op te vragen
        models = client.models.list()

        if not models.models:
            print("⚠️ Geen modellen gevonden. Controleer of je API-sleutel correct is ingesteld.")
            return

        for model in models.models:
            print(f"- {model.name}")

        print("\n==================================================")
        print("✅ Succes! Dit zijn de modellen die jouw account momenteel kan aanroepen.")
        print("==================================================\n")

    except Exception as e:
        print(f"\n❌ Er is een fout opgetreden bij het ophalen van de modellijst: {e}")
        print("Controleer of je genai SDK correct geïnstalleerd is.")

if __name__ == "__main__":
    list_available_models()
