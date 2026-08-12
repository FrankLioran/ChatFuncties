# config.py
# ---------------------------------------------------------
# CENTRALE CONFIGURATIE & PADEN
# ---------------------------------------------------------

from pathlib import Path

# Basispad van het project
BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------
# PADEN EN MAPPEN
# ---------------------------------------------------------
DOCUMENT_FOLDER = BASE_DIR / "Mijn_Documenten"
CHAT_SAVE_DIR = BASE_DIR / "Chats"
OUTPUT_FOLDER_PAINT = BASE_DIR / "EditedImages"
LOG_DIR = BASE_DIR / "Logs"

# Bestandsnamen
INDEX_FILENAME = "document_index.json"
LAZY_INDEX_FILENAME = "document_index_lazy.json"
PROFILE_FILENAME = "eva_profile.json"
LOCAL_SECRETS_PATH = BASE_DIR / "secrets.json"

# ---------------------------------------------------------
# MODEL- EN PROVIDERINSTELLINGEN
# ---------------------------------------------------------
OLLAMA_API_URL = "http://localhost:11434"
DEFAULT_MODEL_NAME = "llama3.2:3b"
DEFAULT_TEMPERATURE = 0.7

# Punt 2: Centraal ingesteld embedding-model
EMBEDDING_MODEL = "mxbai-embed-large:latest"

# ---------------------------------------------------------
# RAG & CONTEXT LIMITS (Punt 15: VRAM-bescherming GTX 1050)
# ---------------------------------------------------------
MAX_CONTEXT_CHARS = 8000       # Maximale contextlengte naar LLM (~2k tokens)
DEFAULT_CHUNK_SIZE = 800       # Karakters per documentchunk
DEFAULT_OVERLAP = 100          # Overlap tussen opeenvolgende chunks

# ---------------------------------------------------------
# OVERIGE INSTELLINGEN
# ---------------------------------------------------------
USER_AGENT = "EvaApp/1.0"

# Zorg dat alle benodigde mappen automatisch bestaan
for folder in [DOCUMENT_FOLDER, CHAT_SAVE_DIR, OUTPUT_FOLDER_PAINT, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)