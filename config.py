# config.py — volledig Hugging Face compatibel
from pathlib import Path
import os

# ---------------------------------------------------------
# 1. Basispad (cross‑platform)
# ---------------------------------------------------------
# Dit werkt op Windows, Linux én Hugging Face Spaces.
BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------
# 2. Projectmappen
# ---------------------------------------------------------
DOCUMENT_FOLDER = BASE_DIR / "Mijn_Documenten"
CHAT_SAVE_DIR = BASE_DIR / "Chats"
OUTPUT_FOLDER_PAINT = BASE_DIR / "EditedImages"
LOG_DIR = BASE_DIR / "Logs"

# Maak alle mappen aan (werkt op HF)
for folder in [DOCUMENT_FOLDER, CHAT_SAVE_DIR, OUTPUT_FOLDER_PAINT, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 3. Bestandsnamen
# ---------------------------------------------------------
INDEX_FILENAME = "document_index.json"
PROFILE_FILENAME = "eva_profile.json"

# ---------------------------------------------------------
# 4. Modelinstellingen
# ---------------------------------------------------------
# Ollama werkt niet op Hugging Face → verwijderen
DEFAULT_MODEL_NAME = "gemini-2.5-flash-lite"
DEFAULT_TEMPERATURE = 0.7

# SentenceTransformers all-MiniLM-L6-v2 heeft embedding-dim 384
EMBEDDING_DIMENSION = 384

USER_AGENT = "EvaApp/1.0"

# ---------------------------------------------------------
# 5. Secrets (streamlit.io)
# ---------------------------------------------------------
# Hugging Face gebruikt environment variables voor secrets.
# Voeg GEMINI_API_KEY toe via: Settings → Secrets → GEMINI_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

