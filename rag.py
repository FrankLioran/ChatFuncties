# rag.py — Hugging Face compatibele versie
import numpy as np
from embeddings import get_embedding_cached
from utils import cosine_similarity

def get_relevant_chunks(question: str, index, top_n=10):
    """
    Vind de meest relevante chunks op basis van SentenceTransformers embeddings.
    Compatibel met Hugging Face (geen Ollama).
    """
    # Vraag-embedding
    q_emb = get_embedding_cached(question)
    scored = []

    for it in index:
        emb_raw = it.get("embedding")

        # Embedding veilig converteren
        if isinstance(emb_raw, list):
            emb = np.array(emb_raw, dtype=np.float32)
        elif isinstance(emb_raw, np.ndarray):
            emb = emb_raw.astype(np.float32)
        else:
            # Fallback: nulvector
            emb = np.zeros_like(q_emb, dtype=np.float32)

        score = cosine_similarity(q_emb, emb)
        scored.append((score, it))

    # Sorteren op relevantie
    scored.sort(key=lambda x: x[0], reverse=True)

    # Top N teruggeven
    return scored[:top_n]
