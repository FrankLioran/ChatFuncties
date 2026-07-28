# rag.py
import numpy as np
from embeddings import get_embedding
from utils import cosine_similarity

def get_relevant_chunks(question: str, index, top_n=10):
    q_emb = get_embedding(question)
    scored = []

    for it in index:
        emb = np.array(it["embedding"], dtype=np.float32)
        score = cosine_similarity(q_emb, emb)
        scored.append((score, it))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_n]
