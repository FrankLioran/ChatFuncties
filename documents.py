# documents.py — Hugging Face compatibele versie
import fitz
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from pathlib import Path
import logging
import json
import glob
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any

# ---------------------------------------------------------
# 1. Embeddings via SentenceTransformers (HF-compatibel)
# ---------------------------------------------------------
from sentence_transformers import SentenceTransformer
import streamlit as st

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource(show_spinner=False)
def get_embedder():
    """
    Laadt het embedding-model één keer.
    Streamlit bewaart het daarna in het geheugen.
    """
    return SentenceTransformer(
        EMBED_MODEL_NAME,
        device="cpu"
    )

def get_embedding_cached(text: str):
    """
    Genereert een embedding.
    Bij een fout wordt een nulvector teruggegeven zodat Eva blijft draaien.
    """
    try:
        embedder = get_embedder()
        return embedder.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    except Exception as e:
        logging.exception(f"Embedding fout: {e}")
        return np.zeros(384, dtype=np.float32)

def cosine_similarity(a, b):
    """Cosine similarity tussen twee vectors."""
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# ---------------------------------------------------------
# 2. Tekst splitting
# ---------------------------------------------------------
def split_text(text: str, chunk_size: int = 800, overlap: int = 100):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = max(0, end - overlap)
    return [c for c in chunks if c]

# ---------------------------------------------------------
# 3. PDF extractie (PyMuPDF)
# ---------------------------------------------------------
def extract_pdf(path: Path):
    """Extractie via PyMuPDF (OCR verwijderd)."""
    try:
        pdf_bytes = path.read_bytes()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = "\n".join(page.get_text() for page in doc)
        chunks = split_text(full_text)
        return full_text, chunks
    except Exception as e:
        logging.exception(f"PDF extractie mislukt: {e}")
        return "", []

# ---------------------------------------------------------
# 4. TXT extractie
# ---------------------------------------------------------
def extract_txt(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = split_text(text)
        return text, chunks
    except Exception as e:
        logging.exception(f"TXT extractie mislukt: {e}")
        return "", []

# ---------------------------------------------------------
# 5. Excel extractie
# ---------------------------------------------------------
def extract_excel(path: Path):
    try:
        df = pd.read_excel(path)
        text = df.to_string(index=False)
        chunks = split_text(text)
        return text, chunks
    except Exception as e:
        logging.exception(f"Excel extractie mislukt: {e}")
        return "", []

# ---------------------------------------------------------
# 6. HTML extractie
# ---------------------------------------------------------
def extract_html(path: Path):
    try:
        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n")
        chunks = split_text(text)
        return text, chunks
    except Exception as e:
        logging.exception(f"HTML extractie mislukt: {e}")
        return "", []

# ---------------------------------------------------------
# 7. On-demand embedding van volledige documenten
# ---------------------------------------------------------
def embed_document_on_demand(source_path):
    try:
        text = Path(source_path).read_text(encoding="utf-8", errors="ignore")
    except:
        return []

    chunks = split_text(text, chunk_size=800, overlap=100)

    embedded_chunks = []
    for ch in chunks:
        emb = get_embedding_cached(ch)
        embedded_chunks.append({
            "content": ch,
            "embedding": emb,
            "source": source_path
        })

    return embedded_chunks

# ---------------------------------------------------------
# 8. Full index
# ---------------------------------------------------------
INDEX_FILENAME = "document_index.json"

def load_or_create_index(folder_path: str):
    index_path = Path(folder_path) / INDEX_FILENAME
    if index_path.is_file():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            st.session_state.document_index = data
            return data
        except Exception:
            logging.exception("Kon index niet laden")
            st.session_state.document_index = []
            return []
    else:
        st.session_state.document_index = []
        return []

def save_index(folder_path: str):
    index_path = Path(folder_path) / INDEX_FILENAME
    try:
        safe = []
        for it in st.session_state.document_index:
            copy = it.copy()
            emb = copy.get("embedding")
            if isinstance(emb, np.ndarray):
                copy["embedding"] = emb.tolist()
            safe.append(copy)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(safe, f, indent=2, ensure_ascii=False)
        return index_path
    except Exception:
        logging.exception("Kon index niet opslaan")
        return None

def scan_and_index_folder_full(folder_path: str):
    folder = Path(folder_path)
    if not folder.is_dir():
        st.sidebar.error(f"Map niet gevonden: {folder}")
        return

    load_or_create_index(folder_path)
    new_entries = []

    supported = ["*.pdf", "*.txt", "*.xlsx", "*.html", "*.htm"]
    files = []
    for pat in supported:
        files.extend(glob.glob(str(folder / pat)))

    for fp in files:
        p = Path(fp)
        try:
            chunks = []
            if p.suffix.lower() == ".pdf":
                _, chunks = extract_pdf(p)
            elif p.suffix.lower() == ".txt":
                _, chunks = extract_txt(p)
            elif p.suffix.lower() == ".xlsx":
                _, chunks = extract_excel(p)
            elif p.suffix.lower() in [".html", ".htm"]:
                _, chunks = extract_html(p)

            for c in chunks:
                emb = get_embedding_cached(c)
                new_entries.append({
                    "content": c,
                    "embedding": emb.tolist(),
                    "source": str(p)
                })

            st.sidebar.info(f"Geïndexeerd (full): {p.name} ({len(chunks)} chunks)")
        except Exception as e:
            st.sidebar.error(f"Fout bij verwerken {p.name}: {e}")

    if new_entries:
        st.session_state.document_index.extend(new_entries)
        save_index(folder_path)
        st.sidebar.success(f"Full index bijgewerkt ({len(new_entries)} nieuwe items).")
    else:
        st.sidebar.info("Geen nieuwe items gevonden.")

# ---------------------------------------------------------
# 9. Full retrieval
# ---------------------------------------------------------
def get_relevant_document_chunks_full(question: str, top_n: int = 10):
    index = st.session_state.get("document_index", [])
    if not index:
        return []

    q_emb = get_embedding_cached(question)
    sims = []

    for it in index:
        emb_raw = it.get("embedding")
        emb = np.array(emb_raw, dtype=np.float32)
        sims.append((cosine_similarity(q_emb, emb), it))

    sims.sort(key=lambda x: x[0], reverse=True)
    top = sims[:top_n]

    return [{"content": it["content"], "source": it["source"]} for score, it in top]

# ---------------------------------------------------------
# 10. Lazy index
# ---------------------------------------------------------
LAZY_INDEX_FILENAME = "document_index_lazy.json"

def summarize_document(text: str, max_chars: int = 1000):
    text = text.strip().replace("\n", " ")
    return text[:max_chars] + ("…" if len(text) > max_chars else "")

def load_or_create_lazy_index(folder_path: str):
    index_path = Path(folder_path) / LAZY_INDEX_FILENAME
    if index_path.is_file():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            st.session_state.document_index_lazy = data
            return data
        except Exception:
            logging.exception("Kon lazy index niet laden")
            st.session_state.document_index_lazy = []
            return []
    else:
        st.session_state.document_index_lazy = []
        return []

def save_lazy_index(folder_path: str):
    index_path = Path(folder_path) / LAZY_INDEX_FILENAME
    try:
        safe = []
        for it in st.session_state.document_index_lazy:
            copy = it.copy()
            emb = copy.get("embedding")
            if isinstance(emb, np.ndarray):
                copy["embedding"] = emb.tolist()
            safe.append(copy)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(safe, f, indent=2, ensure_ascii=False)
        return index_path
    except Exception:
        logging.exception("Kon lazy index niet opslaan")
        return None

def scan_and_index_folder_lazy(folder_path: str):
    folder = Path(folder_path)
    if not folder.is_dir():
        st.sidebar.error(f"Map niet gevonden: {folder}")
        return

    load_or_create_lazy_index(folder_path)
    new_entries = []

    supported = ["*.pdf", "*.txt", "*.xlsx", "*.html", "*.htm"]
    files = []
    for pat in supported:
        files.extend(glob.glob(str(folder / pat)))

    for fp in files:
        p = Path(fp)
        try:
            text_content = ""
            if p.suffix.lower() == ".pdf":
                text_content, _ = extract_pdf(p)
            elif p.suffix.lower() == ".txt":
                text_content, _ = extract_txt(p)
            elif p.suffix.lower() == ".xlsx":
                text_content, _ = extract_excel(p)
            elif p.suffix.lower() in [".html", ".htm"]:
                text_content, _ = extract_html(p)

            summary = summarize_document(text_content)
            emb = get_embedding_cached(summary)

            new_entries.append({
                "filename": p.name,
                "source": str(p),
                "summary": summary,
                "embedding": emb.tolist(),
                "mode": "summary",
                "timestamp": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            })
        except Exception as e:
            logging.exception(f"Fout bij lazy-indexering van {p}: {e}")

    if new_entries:
        st.session_state.document_index_lazy.extend(new_entries)
        save_lazy_index(folder_path)
        st.sidebar.success(f"Lazy index bijgewerkt ({len(new_entries)} nieuwe samenvattingen).")
    else:
        st.sidebar.info("Geen nieuwe documenten gevonden.")

# ---------------------------------------------------------
# 11. Lazy retrieval
# ---------------------------------------------------------
def load_document_on_demand(file_path: str):
    try:
        p = Path(file_path)
        if not p.is_file():
            return []

        if p.suffix.lower() == ".pdf":
            _, chunks = extract_pdf(p)
        elif p.suffix.lower() == ".txt":
            _, chunks = extract_txt(p)
        elif p.suffix.lower() == ".xlsx":
            _, chunks = extract_excel(p)
        elif p.suffix.lower() in [".html", ".htm"]:
            _, chunks = extract_html(p)
        else:
            chunks = []

        return [{"content": c, "source": str(p)} for c in chunks]
    except Exception as e:
        logging.exception(f"Fout bij on-demand laden van {file_path}: {e}")
        return []

def get_relevant_document_chunks_lazy(question: str, top_n_docs: int = 5, top_n_chunks_per_doc: int = 5):
    index = st.session_state.get("document_index_lazy", [])
    if not index:
        return []

    q_emb = get_embedding_cached(question)

    doc_scores = []
    for item in index:
        emb_raw = item.get("embedding")
        emb = np.array(emb_raw, dtype=np.float32)
        score = cosine_similarity(q_emb, emb)
        doc_scores.append((score, item))

    doc_scores.sort(key=lambda x: x[0], reverse=True)
    top_docs = [item for _, item in doc_scores[:top_n_docs]]

    all_chunks = []
    for doc in top_docs:
        chunks = load_document_on_demand(doc["source"])
        all_chunks.extend(chunks)

    if not all_chunks:
        return []

    chunk_scores = []
    for ch in all_chunks:
        emb = get_embedding_cached(ch["content"])
        score = cosine_similarity(q_emb, emb)
        chunk_scores.append((score, ch))

    chunk_scores.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [ch for _, ch in chunk_scores[: top_n_docs * top_n_chunks_per_doc]]

    return top_chunks

# ---------------------------------------------------------
# 12. Context compressie
# ---------------------------------------------------------
def compress_context(chunks: List[Dict[str, str]], max_chars: int = 12000):
    combined = ""
    for ch in chunks:
        block = f"Bron: {ch.get('source','')}\n{ch.get('content','')}\n\n---\n\n"
        if len(combined) + len(block) > max_chars:
            break
        combined += block

    if not combined:
        return "(Geen relevante context gevonden.)"

    return combined

# ---------------------------------------------------------
# 13. Centrale retrieval controller
# ---------------------------------------------------------
def retrieve_context(question: str, mode: str = "auto", top_n: int = 10):
    full_index = st.session_state.get("document_index", [])
    lazy_index = st.session_state.get("document_index_lazy", [])

    # Hybrid
    if mode == "hybrid":
        docs = get_relevant_document_chunks_lazy(question, top_n_docs=5, top_n_chunks_per_doc=1)
        hybrid_chunks = []
        for doc in docs:
            embedded = embed_document_on_demand(doc["source"])
            hybrid_chunks.extend(embedded)

        ranked = sorted(
            hybrid_chunks,
            key=lambda ch: cosine_similarity(get_embedding_cached(question), ch["embedding"]),
            reverse=True
        )[:top_n]

        context = compress_context(ranked)
        st.session_state.last_retrieval_info = {
            "mode": "hybrid",
            "chunks": len(ranked),
            "context_chars": len(context),
        }
        return context

    # Full
    if mode == "full":
        chunks = get_relevant_document_chunks_full(question, top_n=top_n)
        context = compress_context(chunks)
        st.session_state.last_retrieval_info = {
            "mode": "full",
            "chunks": len(chunks),
            "context_chars": len(context),
        }
        return context

    # Lazy
    if mode == "lazy":
        chunks = get_relevant_document_chunks_lazy(question, top_n_docs=5, top_n_chunks_per_doc=3)
        context = compress_context(chunks)
        st.session_state.last_retrieval_info = {
            "mode": "lazy",
            "chunks": len(chunks),
            "context_chars": len(context),
        }
        return context

    # Auto
    if mode == "auto":
        if full_index and len(full_index) < 5000:
            chunks = get_relevant_document_chunks_full(question, top_n=top_n)
            context = compress_context(chunks)
            st.session_state.last_retrieval_info = {
                "mode": "auto/full",
                "chunks": len(chunks),
                "context_chars": len(context),
            }
            return context

        if lazy_index:
            docs = get_relevant_document_chunks_lazy(question, top_n_docs=5, top_n_chunks_per_doc=1)
            hybrid_chunks = []
            for doc in docs:
                embedded = embed_document_on_demand(doc["source"])
                hybrid_chunks.extend(embedded)

            ranked = sorted(
                hybrid_chunks,
                key=lambda ch: cosine_similarity(get_embedding_cached(question), ch["embedding"]),
                reverse=True
            )[:top_n]

            context = compress_context(ranked)
            st.session_state.last_retrieval_info = {
                "mode": "auto/hybrid",
                "chunks": len(ranked),
                "context_chars": len(context),
            }
            return context

    return "(Geen context beschikbaar.)"
