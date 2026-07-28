# config.py
from pathlib import Path

# Basispad voor deze nieuwe versie
BASE_DIR = Path(__file__).resolve().parent

# Paden
DOCUMENT_FOLDER = BASE_DIR / "Mijn_Documenten"
CHAT_SAVE_DIR = BASE_DIR / "Chats"
OUTPUT_FOLDER_PAINT = BASE_DIR / "EditedImages"
LOG_DIR = BASE_DIR / "Logs"

# Bestanden
INDEX_FILENAME = "document_index.json"
PROFILE_FILENAME = "eva_profile.json"

# Modelinstellingen
OLLAMA_API_URL = "http://localhost:11434"
DEFAULT_MODEL_NAME = "llama3.2:3b"
DEFAULT_TEMPERATURE = 0.7
EMBEDDING_DIMENSION = 768

USER_AGENT = "EvaApp/1.0"
LOCAL_SECRETS_PATH = BASE_DIR / "secrets.json"

# Zorg dat mappen bestaan
for folder in [DOCUMENT_FOLDER, CHAT_SAVE_DIR, OUTPUT_FOLDER_PAINT, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
