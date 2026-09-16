# documents.py
# ---------------------------------------------------------
# RAG DOCUMENT PARSER & RETRIEVAL SYSTEM
#
# - TXT / PDF / DOCX / XLSX / HTML
# - OCR fallback voor PDF
# - full index
# - lazy index
# - batch embedding
# - semantische ranking
# - keyword fallback
# ---------------------------------------------------------

import io
import re
import json
import glob
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional

import pymupdf
import docx
import pandas as pd
import numpy as np
import pytesseract

try:
    from pdf2image import convert_from_path, pdfinfo_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from bs4 import BeautifulSoup
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from embeddings import (
    get_embedding_cached,
    get_embeddings_batch,
)
from utils import cosine_similarity, split_text

try:
    from config import EMBEDDING_MODEL as DEFAULT_EMBED_MODEL
except ImportError:
    DEFAULT_EMBED_MODEL = "mxbai-embed-large:latest"


# ---------------------------------------------------------
# 1. CONFIGURATIE
# ---------------------------------------------------------

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 100

# Aantal chunks per embedding-batch.
# Dit is bewust bescheiden gehouden voor RAM/API-belasting.
EMBED_BATCH_SIZE = 16

MAX_CONTEXT_CHARS = 8000
MAX_CHUNK_CHARS_IN_CONTEXT = 1800
MAX_EMBED_CHUNKS = 500
SUMMARY_LENGTH = 1000

TOP_DOCS = 5
TOP_CHUNKS = 5
DEFAULT_TOP_N = 8

INDEX_FILENAME = "document_index.json"
LAZY_INDEX_FILENAME = "document_index_lazy.json"

if Path("/usr/bin/tesseract").exists():
    pytesseract.pytesseract.tesseract_cmd = (
        "/usr/bin/tesseract"
    )


# ---------------------------------------------------------
# 2. DOCUMENT PARSER & EXTRACTIE
# ---------------------------------------------------------

def load_document(
    file: Union[Path, str, UploadedFile],
) -> str:
    """
    Universele documentlezer.

    Ondersteunt:
        TXT
        PDF
        DOCX
        XLSX / XLS
        HTML / HTM
    """

    if isinstance(file, UploadedFile):
        filename = file.name.lower()

        content_bytes = (
            file.getvalue()
            if hasattr(file, "getvalue")
            else file.read()
        )

    else:
        p = Path(file)
        filename = p.name.lower()
        content_bytes = None

    suffix = Path(filename).suffix.lower()

    # TXT
    if suffix == ".txt":
        if content_bytes is not None:
            return content_bytes.decode(
                "utf-8",
                errors="ignore",
            )

        return Path(file).read_text(
            encoding="utf-8",
            errors="ignore",
        )

    # PDF
    elif suffix == ".pdf":
        if content_bytes is not None:
            pdf = pymupdf.open(
                stream=content_bytes,
                filetype="pdf",
            )
        else:
            pdf = pymupdf.open(str(file))

        try:
            return "\n".join(
                page.get_text()
                for page in pdf
            )
        finally:
            pdf.close()

    # DOCX
    elif suffix == ".docx":
        if content_bytes is not None:
            doc = docx.Document(
                io.BytesIO(content_bytes)
            )
        else:
            doc = docx.Document(str(file))

        return "\n".join(
            p.text
            for p in doc.paragraphs
        )

    # XLSX / XLS
    elif suffix in (".xlsx", ".xls"):
        if content_bytes is not None:
            df = pd.read_excel(
                io.BytesIO(content_bytes)
            )
        else:
            df = pd.read_excel(str(file))

        return df.to_string(index=False)

    # HTML / HTM
    elif suffix in (".html", ".htm"):
        if content_bytes is not None:
            html_str = content_bytes.decode(
                "utf-8",
                errors="ignore",
            )
        else:
            html_str = Path(file).read_text(
                encoding="utf-8",
                errors="ignore",
            )

        soup = BeautifulSoup(
            html_str,
            "html.parser",
        )

        return soup.get_text(
            separator="\n"
        )

    raise ValueError(
        f"Bestandstype niet ondersteund: {filename}"
    )


def extract_pdf_ocr(path: Path) -> str:
    """
    Pagina-voor-pagina OCR-extractie.
    """

    if not OCR_AVAILABLE:
        logging.warning(
            "OCR niet beschikbaar "
            "(pdf2image ontbreekt)."
        )
        return ""

    try:
        info = pdfinfo_from_path(path)
        total_pages = info.get(
            "Pages",
            0,
        )

        text = ""

        for page_num in range(
            1,
            total_pages + 1,
        ):
            images = convert_from_path(
                path,
                first_page=page_num,
                last_page=page_num,
            )

            for img in images:
                text += (
                    pytesseract.image_to_string(img)
                    + "\n"
                )

        return text

    except Exception:
        logging.exception(
            "OCR extractie mislukt voor %s",
            path,
        )
        return ""


def extract_document(
    path: Path,
) -> Tuple[str, List[str]]:
    """
    Centrale documentparser.
    """

    try:
        text = load_document(path)

        if (
            path.suffix.lower() == ".pdf"
            and not text.strip()
        ):
            logging.info(
                "Geen directe tekst in PDF '%s', "
                "OCR gestart.",
                path.name,
            )
            text = extract_pdf_ocr(path)

        chunks = split_text(
            text,
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap=DEFAULT_OVERLAP,
        )

        return text, chunks

    except Exception as exc:
        logging.exception(
            "Fout bij verwerken van document %s: %s",
            path,
            exc,
        )
        return "", []


# ---------------------------------------------------------
# 3. BATCH EMBEDDING HELPER
# ---------------------------------------------------------

def _batch_process_chunks(
    chunks: List[str],
    embed_model: str,
    source_path: str = "",
) -> List[Dict[str, Any]]:
    """
    Zet chunks om naar records met embeddings.

    De embeddinglaag garandeert dat vectors dezelfde lengte
    en volgorde hebben als de inputbatch.
    """

    results: List[Dict[str, Any]] = []

    for i in range(
        0,
        len(chunks),
        EMBED_BATCH_SIZE,
    ):
        batch = chunks[
            i:i + EMBED_BATCH_SIZE
        ]

        vectors = get_embeddings_batch(
            batch,
            model=embed_model,
        )

        # Extra defensieve controle.
        if len(vectors) != len(batch):
            logger_message = (
                "Embedding batch had onverwachte lengte: "
                f"chunks={len(batch)}, "
                f"vectors={len(vectors)}"
            )
            logging.error(
                logger_message
            )

            # Maak de mapping expliciet veilig.
            vectors = list(vectors[:len(batch)])

            if len(vectors) < len(batch):
                vectors.extend(
                    [None]
                    * (len(batch) - len(vectors))
                )

        for chunk_text, vec in zip(
            batch,
            vectors,
        ):
            results.append(
                {
                    "content": chunk_text,
                    "embedding": (
                        np.asarray(
                            vec,
                            dtype=np.float32,
                        )
                        if vec is not None
                        else None
                    ),
                    "source": source_path,
                }
            )

    return results


# ---------------------------------------------------------
# 4. ON-DEMAND EMBEDDING
# ---------------------------------------------------------

def embed_document_on_demand(
    source_path: str,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> List[Dict[str, Any]]:
    """
    Laadt een document, maakt chunks en embedt deze in batches.
    """

    try:
        text = load_document(
            Path(source_path)
        )
    except Exception:
        logging.exception(
            "Kon document niet laden: %s",
            source_path,
        )
        return []

    chunks = split_text(
        text,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_OVERLAP,
    )[:MAX_EMBED_CHUNKS]

    return _batch_process_chunks(
        chunks,
        embed_model=embed_model,
        source_path=str(source_path),
    )


# ---------------------------------------------------------
# 5. FULL INDEX
# ---------------------------------------------------------

def load_or_create_index(
    folder_path: str,
) -> List[Dict[str, Any]]:
    index_path = (
        Path(folder_path)
        / INDEX_FILENAME
    )

    if index_path.is_file():
        try:
            with open(
                index_path,
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            st.session_state.document_index = data
            return data

        except Exception:
            logging.exception(
                "Kon index niet laden"
            )

    st.session_state.document_index = []
    return []


def save_index(
    folder_path: str,
) -> Optional[Path]:
    index_path = (
        Path(folder_path)
        / INDEX_FILENAME
    )

    try:
        safe = []

        for item in st.session_state.get(
            "document_index",
            [],
        ):
            copy = item.copy()

            emb = copy.get(
                "embedding"
            )

            if isinstance(
                emb,
                np.ndarray,
            ):
                copy["embedding"] = (
                    emb.tolist()
                )

            safe.append(copy)

        with open(
            index_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                safe,
                f,
                indent=2,
                ensure_ascii=False,
            )

        return index_path

    except Exception:
        logging.exception(
            "Kon index niet opslaan"
        )
        return None


def scan_and_index_folder_full(
    folder_path: str,
    embed_model: str = DEFAULT_EMBED_MODEL,
):
    folder = Path(folder_path)

    if not folder.is_dir():
        st.sidebar.error(
            f"Map niet gevonden: {folder}"
        )
        return

    load_or_create_index(
        folder_path
    )

    current_index = (
        st.session_state.get(
            "document_index",
            [],
        )
    )

    # Per bron de nieuwste mtime bewaren.
    indexed_map = {}

    for item in current_index:
        src = item.get("source")
        mtime = item.get(
            "mtime",
            0,
        )

        if src:
            indexed_map[src] = max(
                indexed_map.get(src, 0),
                mtime,
            )

    supported = [
        "*.pdf",
        "*.txt",
        "*.docx",
        "*.xlsx",
        "*.html",
        "*.htm",
    ]

    files: List[str] = []

    for pattern in supported:
        files.extend(
            glob.glob(
                str(folder / pattern)
            )
        )

    new_entries: List[
        Dict[str, Any]
    ] = []

    updated_sources = set()

    for fp in files:
        p = Path(fp)
        mtime = p.stat().st_mtime
        str_p = str(p)

        if (
            str_p in indexed_map
            and indexed_map[str_p] >= mtime
        ):
            continue

        try:
            updated_sources.add(
                str_p
            )

            _, chunks = extract_document(p)

            embedded_records = (
                _batch_process_chunks(
                    chunks,
                    embed_model=embed_model,
                    source_path=str_p,
                )
            )

            for record in embedded_records:
                emb = record["embedding"]

                new_entries.append(
                    {
                        "content": record["content"],
                        "embedding": (
                            emb.tolist()
                            if isinstance(
                                emb,
                                np.ndarray,
                            )
                            else emb
                        ),
                        "source": str_p,
                        "mtime": mtime,
                    }
                )

            logging.info(
                "Geïndexeerd (full, batch): "
                "%s (%d chunks)",
                p.name,
                len(chunks),
            )

        except Exception:
            logging.exception(
                "Fout bij verwerken %s",
                p.name,
            )

    if updated_sources:
        cleaned_index = [
            item
            for item in current_index
            if item.get("source")
            not in updated_sources
        ]

        cleaned_index.extend(
            new_entries
        )

        st.session_state.document_index = (
            cleaned_index
        )

        save_index(folder_path)

        st.sidebar.success(
            "Full index bijgewerkt "
            f"({len(new_entries)} items)."
        )

    else:
        st.sidebar.info(
            "Geen nieuwe of gewijzigde "
            "bestanden gevonden."
        )


def get_relevant_document_chunks_full(
    question: str,
    top_n: int = DEFAULT_TOP_N,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> List[Dict[str, str]]:
    index = st.session_state.get(
        "document_index",
        [],
    )

    if not index:
        return []

    q_emb = get_embedding_cached(
        question,
        model=embed_model,
    )

    if q_emb is None:
        logging.warning(
            "Vraag-embedding kon niet "
            "berekend worden."
        )
        return []

    sims: List[
        Tuple[
            float,
            Dict[str, Any],
        ]
    ] = []

    for item in index:
        emb_raw = item.get(
            "embedding"
        )

        if emb_raw is None:
            continue

        emb = np.asarray(
            emb_raw,
            dtype=np.float32,
        )

        try:
            score = cosine_similarity(
                q_emb,
                emb,
            )
        except Exception:
            logging.warning(
                "Embeddingdimensie komt "
                "niet overeen voor %s.",
                item.get("source"),
            )
            continue

        sims.append(
            (
                score,
                item,
            )
        )

    sims.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        {
            "content": item.get(
                "content",
                "",
            ),
            "source": item.get(
                "source",
                "",
            ),
        }
        for _, item in sims[:top_n]
    ]


# ---------------------------------------------------------
# 6. LAZY INDEX
# ---------------------------------------------------------

def summarize_document(
    text: str,
    max_chars: int = SUMMARY_LENGTH,
) -> str:
    text = (
        text
        .strip()
        .replace("\n", " ")
    )

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "…"


def load_or_create_lazy_index(
    folder_path: str,
) -> List[Dict[str, Any]]:
    index_path = (
        Path(folder_path)
        / LAZY_INDEX_FILENAME
    )

    if index_path.is_file():
        try:
            with open(
                index_path,
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            st.session_state.document_index_lazy = (
                data
            )

            return data

        except Exception:
            logging.exception(
                "Kon lazy index niet laden"
            )

    st.session_state.document_index_lazy = []
    return []


def save_lazy_index(
    folder_path: str,
) -> Optional[Path]:
    index_path = (
        Path(folder_path)
        / LAZY_INDEX_FILENAME
    )

    try:
        safe = []

        for item in st.session_state.get(
            "document_index_lazy",
            [],
        ):
            copy = item.copy()

            emb = copy.get(
                "embedding"
            )

            if isinstance(
                emb,
                np.ndarray,
            ):
                copy["embedding"] = (
                    emb.tolist()
                )

            safe.append(copy)

        with open(
            index_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                safe,
                f,
                indent=2,
                ensure_ascii=False,
            )

        return index_path

    except Exception:
        logging.exception(
            "Kon lazy index niet opslaan"
        )
        return None


def scan_and_index_folder_lazy(
    folder_path: str,
    embed_model: str = DEFAULT_EMBED_MODEL,
):
    folder = Path(folder_path)

    if not folder.is_dir():
        st.sidebar.error(
            f"Map niet gevonden: {folder}"
        )
        return

    load_or_create_lazy_index(
        folder_path
    )

    current_lazy = (
        st.session_state.get(
            "document_index_lazy",
            [],
        )
    )

    indexed_map = {
        item["source"]: item.get(
            "mtime",
            0,
        )
        for item in current_lazy
        if "source" in item
    }

    supported = [
        "*.pdf",
        "*.txt",
        "*.docx",
        "*.xlsx",
        "*.html",
        "*.htm",
    ]

    files: List[str] = []

    for pattern in supported:
        files.extend(
            glob.glob(
                str(folder / pattern)
            )
        )

    new_entries: List[
        Dict[str, Any]
    ] = []

    updated_sources = set()

    for fp in files:
        p = Path(fp)
        mtime = p.stat().st_mtime
        str_p = str(p)

        if (
            str_p in indexed_map
            and indexed_map[str_p] >= mtime
        ):
            continue

        try:
            text_content, _ = (
                extract_document(p)
            )

            if not text_content.strip():
                continue

            summary = summarize_document(
                text_content
            )

            emb = get_embedding_cached(
                summary,
                model=embed_model,
            )

            new_entries.append(
                {
                    "filename": p.name,
                    "source": str_p,
                    "summary": summary,
                    "embedding": (
                        emb.tolist()
                        if isinstance(
                            emb,
                            np.ndarray,
                        )
                        else emb
                    ),
                    "mode": "summary",
                    "mtime": mtime,
                    "timestamp":
                        datetime.fromtimestamp(
                            mtime
                        ).isoformat(),
                }
            )

            updated_sources.add(
                str_p
            )

        except Exception:
            logging.exception(
                "Fout bij lazy-indexering "
                "van %s",
                p,
            )

    if updated_sources:
        cleaned_lazy = [
            item
            for item in current_lazy
            if item.get("source")
            not in updated_sources
        ]

        cleaned_lazy.extend(
            new_entries
        )

        st.session_state.document_index_lazy = (
            cleaned_lazy
        )

        save_lazy_index(
            folder_path
        )

        st.sidebar.success(
            "Lazy index bijgewerkt "
            f"({len(new_entries)} items)."
        )

    else:
        st.sidebar.info(
            "Geen nieuwe of gewijzigde "
            "bestanden gevonden."
        )


def get_relevant_document_chunks_lazy(
    question: str,
    top_n_docs: int = TOP_DOCS,
    top_n_chunks_per_doc: int = TOP_CHUNKS,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> List[Dict[str, str]]:
    index = st.session_state.get(
        "document_index_lazy",
        [],
    )

    if not index:
        return []

    q_emb = get_embedding_cached(
        question,
        model=embed_model,
    )

    if q_emb is None:
        return []

    doc_scores: List[
        Tuple[
            float,
            Dict[str, Any],
        ]
    ] = []

    for item in index:
        emb_raw = item.get(
            "embedding"
        )

        if emb_raw is None:
            continue

        emb = np.asarray(
            emb_raw,
            dtype=np.float32,
        )

        try:
            score = cosine_similarity(
                q_emb,
                emb,
            )
        except Exception:
            logging.warning(
                "Embeddingdimensie komt "
                "niet overeen voor %s.",
                item.get("source"),
            )
            continue

        doc_scores.append(
            (
                score,
                item,
            )
        )

    doc_scores.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    top_docs = [
        item
        for _, item in doc_scores[:top_n_docs]
    ]

    all_chunks: List[
        Dict[str, Any]
    ] = []

    for doc in top_docs:
        chunks_embedded = (
            embed_document_on_demand(
                doc["source"],
                embed_model=embed_model,
            )
        )

        all_chunks.extend(
            chunks_embedded
        )

    if not all_chunks:
        return []

    chunk_scores: List[
        Tuple[
            float,
            Dict[str, Any],
        ]
    ] = []

    for chunk in all_chunks:
        emb = chunk.get(
            "embedding"
        )

        if emb is None:
            continue

        try:
            score = cosine_similarity(
                q_emb,
                emb,
            )
        except Exception:
            continue

        chunk_scores.append(
            (
                score,
                chunk,
            )
        )

    chunk_scores.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        {
            "content": chunk["content"],
            "source": chunk["source"],
        }
        for _, chunk in chunk_scores[
            : top_n_docs * top_n_chunks_per_doc
        ]
    ]


# ---------------------------------------------------------
# 7. RANKING
# ---------------------------------------------------------

def rank_chunks_by_keyword(
    question: str,
    chunks: List[Dict[str, Any]],
    top_n: int = DEFAULT_TOP_N,
) -> List[Dict[str, Any]]:
    """
    Keyword/Jaccard fallback wanneer embeddings niet beschikbaar zijn.
    """

    q_words = set(
        re.findall(
            r"\w+",
            question.lower(),
        )
    )

    if not q_words:
        return chunks[:top_n]

    scored = []

    for chunk in chunks:
        content = chunk.get(
            "content",
            chunk.get(
                "summary",
                "",
            ),
        )

        c_words = set(
            re.findall(
                r"\w+",
                content.lower(),
            )
        )

        overlap = len(
            q_words.intersection(
                c_words
            )
        )

        score = (
            overlap
            / (
                len(q_words)
                + len(c_words)
                - overlap
                + 1e-5
            )
        )

        scored.append(
            (
                score,
                chunk,
            )
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        chunk
        for _, chunk in scored[:top_n]
    ]


def rank_chunks(
    question: str,
    chunks: List[Dict[str, Any]],
    top_n: int = DEFAULT_TOP_N,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> List[Dict[str, Any]]:
    """
    Semantische ranking op basis van cosine similarity.
    """

    if not chunks:
        return []

    q_emb = get_embedding_cached(
        question,
        model=embed_model,
    )

    if q_emb is None:
        logging.warning(
            "Geen vraag-embedding beschikbaar; "
            "valt terug op keyword ranking."
        )

        return rank_chunks_by_keyword(
            question,
            chunks,
            top_n=top_n,
        )

    scored = []

    for chunk in chunks:
        emb_raw = chunk.get(
            "embedding"
        )

        if emb_raw is None:
            continue

        emb = np.asarray(
            emb_raw,
            dtype=np.float32,
        )

        try:
            score = cosine_similarity(
                q_emb,
                emb,
            )
        except Exception:
            continue

        scored.append(
            (
                score,
                chunk,
            )
        )

    # Als geen enkele embedding compatibel is,
    # alsnog bruikbaar terugvallen.
    if not scored:
        logging.warning(
            "Geen compatibele chunk-embeddings gevonden; "
            "valt terug op keyword ranking."
        )

        return rank_chunks_by_keyword(
            question,
            chunks,
            top_n=top_n,
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        chunk
        for _, chunk in scored[:top_n]
    ]


# ---------------------------------------------------------
# 8. CONTEXT COMPRESSIE
# ---------------------------------------------------------

def compress_context(
    chunks: List[Dict[str, str]],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Combineert geselecteerde chunks tot één contextblok.
    """

    combined = ""

    for chunk in chunks:
        content = chunk.get(
            "content",
            "",
        )[:MAX_CHUNK_CHARS_IN_CONTEXT]

        source = chunk.get(
            "source",
            "Onbekende bron",
        )

        block = (
            f"Bron: {source}\n"
            f"{content}\n\n"
            "---\n\n"
        )

        if (
            len(combined)
            + len(block)
            > max_chars
        ):
            break

        combined += block

    if not combined:
        return (
            "(Geen relevante "
            "documentcontext gevonden.)"
        )

    return combined


# ---------------------------------------------------------
# 9. CENTRALE RETRIEVAL CONTROLLER
# ---------------------------------------------------------

def retrieve_context(
    question: str,
    mode: str = "auto",
    top_n: int = DEFAULT_TOP_N,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> str:
    """
    Centrale regisseur voor document retrieval.

    Ondersteunt:
        - direct geüploade documenten
        - full index
        - lazy index
        - semantic retrieval
        - keyword fallback
    """

    logging.info(
        "retrieve_context gestart"
    )

    uploaded_sections = (
        st.session_state.get(
            "sections",
            [],
        )
    )

    full_index = (
        st.session_state.get(
            "document_index",
            [],
        )
    )

    lazy_index = (
        st.session_state.get(
            "document_index_lazy",
            [],
        )
    )

    # Controleer embedding-service.
    test_emb = get_embedding_cached(
        "test",
        model=embed_model,
    )

    embeddings_offline = (
        test_emb is None
        or len(test_emb) == 0
    )

    retrieved_chunks: List[
        Dict[str, Any]
    ] = []

    retrieval_mode = "none"

    effective_mode = st.session_state.get(
        "rag_mode",
        mode,
    )

    if effective_mode not in (
        "auto",
        "semantic",
        "keyword",
    ):
        effective_mode = "auto"

    # -----------------------------------------------------
    # 1. DIRECT GEÜPLOAD DOCUMENT
    # -----------------------------------------------------

    if uploaded_sections:
        chunks_to_rank = [
            {
                "content": chunk,
                "source":
                    "Direct Geüpload Document",
            }
            for chunk in uploaded_sections
        ]

        if (
            embeddings_offline
            or effective_mode == "keyword"
        ):
            retrieved_chunks = (
                rank_chunks_by_keyword(
                    question,
                    chunks_to_rank,
                    top_n=top_n,
                )
            )

            retrieval_mode = (
                "uploaded_doc_keyword"
            )

        else:
            embedded_list = (
                _batch_process_chunks(
                    uploaded_sections,
                    embed_model=embed_model,
                    source_path=(
                        "Direct Geüpload Document"
                    ),
                )
            )

            retrieved_chunks = rank_chunks(
                question,
                embedded_list,
                top_n=top_n,
                embed_model=embed_model,
            )

            retrieval_mode = (
                "uploaded_doc_semantic"
            )

    # -----------------------------------------------------
    # 2. FULL INDEX
    # -----------------------------------------------------

    elif (
        full_index
        and effective_mode
        in ("auto", "semantic", "keyword")
    ):
        if (
            embeddings_offline
            or effective_mode == "keyword"
        ):
            retrieved_chunks = (
                rank_chunks_by_keyword(
                    question,
                    full_index,
                    top_n=top_n,
                )
            )

            retrieval_mode = (
                "full_index_keyword"
            )

        else:
            retrieved_chunks = (
                get_relevant_document_chunks_full(
                    question,
                    top_n=top_n,
                    embed_model=embed_model,
                )
            )

            retrieval_mode = (
                "full_index_semantic"
            )

    # -----------------------------------------------------
    # 3. LAZY INDEX
    # -----------------------------------------------------

    elif (
        lazy_index
        and effective_mode
        in ("auto", "semantic", "keyword")
    ):
        if (
            embeddings_offline
            or effective_mode == "keyword"
        ):
            retrieved_chunks = (
                rank_chunks_by_keyword(
                    question,
                    lazy_index,
                    top_n=top_n,
                )
            )

            retrieval_mode = (
                "lazy_index_keyword"
            )

        else:
            retrieved_chunks = (
                get_relevant_document_chunks_lazy(
                    question,
                    embed_model=embed_model,
                )
            )

            retrieval_mode = (
                "lazy_index_semantic"
            )

    context_str = compress_context(
        retrieved_chunks
    )

    st.session_state.last_retrieval_info = {
        "mode": retrieval_mode,
        "chunks": len(
            retrieved_chunks
        ),
        "context_chars": len(
            context_str
        ),
        "preview": (
            context_str[:1500]
            + (
                "…"
                if len(context_str) > 1500
                else ""
            )
        ),
    }

    logging.info(
        "retrieve_context voltooid "
        "(%d chunks via '%s')",
        len(retrieved_chunks),
        retrieval_mode,
    )

    return context_str


# ---------------------------------------------------------
# 10. DOCUMENT → CHUNKS
# ---------------------------------------------------------

def document_to_chunks(
    file: Union[
        Path,
        str,
        UploadedFile,
    ],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
    text = load_document(file)

    return split_text(
        text,
        chunk_size,
        overlap,
    )

