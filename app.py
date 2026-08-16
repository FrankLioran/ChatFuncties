import uuid
import random
from pathlib import Path
import streamlit as st

from config import DOCUMENT_FOLDER, DEFAULT_MODEL_NAME, DEFAULT_TEMPERATURE
from chat import answer_question, load_profile, get_persona_image
from audio import listen_and_transcribe
from documents import load_document
from image import generate_and_save_image
from utils import save_chat_to_txt, split_text


def get_welcome_line() -> str:
    return random.choice([
        "Welkom 🌿 — ik ben Eva. Waarmee kan ik helpen?",
        "Hoi — ik help je graag rustig en helder.",
        "Hallo — vraag maar, ik denk met je mee."
    ])

# ---------------------------------------------------------
# Pagina Configuratie & Initialisatie
# ---------------------------------------------------------
st.set_page_config(page_title="Eva — Vraag & Antwoord", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": get_welcome_line()
    }]

if "user_data" not in st.session_state:
    st.session_state.user_data = {}

if st.session_state.session_id not in st.session_state.user_data:
    st.session_state.user_data[st.session_state.session_id] = {
        "messages": [],
        "image": None,
        "persona": None,
        "profile": None
    }

defaults = {
    "model_name": DEFAULT_MODEL_NAME,
    "temperature": DEFAULT_TEMPERATURE,
    "document_folder": str(DOCUMENT_FOLDER),
    "document_index": [],
    "pdf_text": None,
    "rag_mode": "auto",  # Geldige opties: "auto", "semantic", "keyword"
    "sections": [],
    "section_embeddings": [],
    "spoken_text_input": "",
    "spoken_processed": False,
    "persona": None,
    "profile": None,
    "excel_df": None,
    "web_context": "",
    "gemini_api_key_user": "",
    "groq_api_key_user": "",
    "safe_mode": True,
    "token_limit": 200000,
    "tokens_used": 0,
    "request_limit": 100,
    "requests_used": 0,
    "last_retrieval_info": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------------------------------------------------
# Sidebar UI & Instellingen
# ---------------------------------------------------------
with st.sidebar:
    st.image(str(get_persona_image()), width=220)

    st.header("Persona")
    persona_choice = st.selectbox(
        "Kies een persona",
        ["Eva Lumen", "Astraea", "Standaard"],
        index=0
    )
    st.session_state["active_persona"] = persona_choice

    persona_content, profile_text = load_profile()
    st.session_state.persona = persona_content
    st.session_state.profile = profile_text

    st.header("Modelinstellingen")
    st.session_state.temperature = st.slider(
        "Temperature",
        0.0, 1.5,
        float(st.session_state.temperature)
    )

    st.header("AI Provider")
    st.session_state["ai_provider"] = st.selectbox(
        "Kies een AI-provider",
        ["Lokaal", "Gemini", "Groq"],
        index=0
    )

    if st.session_state["ai_provider"] == "Lokaal":
        st.session_state.model_name = st.selectbox(
            "Modelnaam",
            ["qwen3.5", "gemma4:e4b", "gemma4:e2b", "gemma3:4b", "gemma2:9B", "qwen2.5:3b", "llama3.2:3b", "gemma3:1b", "mistral:7b",],
            index=0
        )
    
    elif st.session_state["ai_provider"] == "Gemini":
        st.session_state.model_name = st.selectbox(
            "Modelnaam",
            [
                "gemini-3.6-flash",
                "gemma-4-26b-a4b-it",
                "gemini-2.5-flash-lite",
                "gemini-3.5-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite",
                "gemma-4-31b-it"
            ],
            index=0
        )
    elif st.session_state["ai_provider"] == "Groq":
        st.session_state.model_name = st.selectbox(
            "Modelnaam",
            ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"],
            index=0
        )

    # Veiligheid & Limieten
    st.subheader("🛡️ Veiligheid")
    st.session_state.safe_mode = st.checkbox(
        "Safe Mode (aanbevolen)",
        value=st.session_state.safe_mode
    )

    if st.session_state.safe_mode:
        st.session_state.token_limit = st.number_input(
            "Maximum tokens per sessie",
            min_value=1000, max_value=1000000, value=200000, step=1000
        )
        st.session_state.request_limit = st.number_input(
            "Maximum aantal requests",
            min_value=1, max_value=10000, value=100
        )
    else:
        st.warning("Safe Mode is uitgeschakeld.")

    st.progress(min(st.session_state.tokens_used / max(st.session_state.token_limit, 1), 1.0))
    st.caption(f"Gebruikte tokens: {st.session_state.tokens_used:,}")

    # Document Upload
    st.subheader("📄 Document upload")
    uploaded_file = st.file_uploader(
        "Upload een document",
        type=["pdf", "txt", "docx", "html", "htm"]
    )

    if uploaded_file is not None:
        try:
            text = load_document(uploaded_file)
            chunks = split_text(text, chunk_size=800, overlap=100)

            st.session_state.pdf_text = text
            st.session_state.sections = chunks
            st.session_state.section_embeddings = []  # Reset embeddings bij nieuw document
            st.success(f"✅ {uploaded_file.name} geladen ({len(chunks)} chunks)")
        except Exception as e:
            st.error(f"Kon document niet laden:\n{e}")

    # RAG modus selectie (Punt 11 opgelost)
    st.subheader("🔍 Retrieval Modus")
    st.session_state.rag_mode = st.selectbox(
        "RAG Zoekmethode",
        ["auto", "semantic", "keyword"],
        index=0,
        help="'auto' kiest de beste methode; 'semantic' gebruikt vector-embeddings."
    )

    # Afbeeldingen & Spraak
    st.header("🎙️ Spraak")
    if st.button("Luister naar spraak"):
        text = listen_and_transcribe()
        if text:
            st.session_state.spoken_text_input = text

    st.header("🎨 Afbeeldingen")
    image_prompt = st.text_input("Omschrijving afbeelding:", value="portrait of a red rose in rain")
    if st.button("Genereer afbeelding"):
        with st.spinner("Afbeelding wordt gegenereerd..."):
            path = generate_and_save_image(image_prompt)
            if path and Path(path).exists():
                with open(path, "rb") as f:
                    img_bytes = f.read()
                st.session_state.messages.append({
                    "role": "image",
                    "content": f"🎨 Afbeelding gegenereerd voor: *'{image_prompt}'*",
                    "image_bytes": img_bytes
                })
                st.rerun()
            else:
                st.error("Genereren van afbeelding mislukt.")

    # Debugpaneel (Punt 10 & 15 opgelost)
    st.subheader("🐞 Debugpaneel")
    if st.checkbox("Toon geüploade chunks", key="show_raw_chunks"):
        if st.session_state.get("sections"):
            st.caption(f"Totaal aantal chunks: {len(st.session_state.sections)}")
            chunk_idx = st.number_input("Chunk index", min_value=0, max_value=max(0, len(st.session_state.sections)-1), step=1)
            st.text_area("Chunk inhoud", st.session_state.sections[chunk_idx], height=150)
        else:
            st.info("Geen chunks beschikbaar.")

    if st.session_state.get("last_retrieval_info"):
        info = st.session_state.last_retrieval_info
        st.write(f"**Retrieval Modus:** {info.get('mode', 'onbekend')}")
        st.write(f"**Gevonden chunks:** {info.get('chunks', 0)}")

        ctx_len = info.get("context_chars", 0)
        st.write(f"**Contextlengte:** {ctx_len:,} tekens")

        # VRAM Geheugenindicatie voor GTX 1050 (~8k limiet)
        if ctx_len < 4000:
            st.success("Contextgrootte: Optimaal 🟢")
        elif ctx_len <= 8000:
            st.warning("Contextgrootte: Aanvaardbaar 🟠")
        else:
            st.error("Contextgrootte: Te groot voor GTX 1050 🔴")

        if "preview" in info and info["preview"]:
            with st.expander("👁️ Exacte Context Preview naar AI"):
                st.text_area("Context sent to LLM", info["preview"], height=200)
    else:
        st.info("Nog geen retrieval uitgevoerd.")

    # Reset & Opslaan
    st.markdown("---")
    if st.button("🔄 Reset sessie"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": get_welcome_line()
        }]
        st.session_state.pdf_text = None
        st.session_state.sections = []
        st.session_state.section_embeddings = []
        st.session_state.last_retrieval_info = None
        st.rerun()

    if st.button("📥 Bewaar gesprek", key="save_chat_button"):
        if st.session_state.get("messages"):
            fp = save_chat_to_txt(st.session_state.messages)
            if fp:
                st.sidebar.success(f"✅ Bewaard als {fp.name}")
                with open(fp, "rb") as f:
                    st.sidebar.download_button("⬇️ Download file", data=f, file_name=fp.name, key="dl_chat")
            else:
                st.sidebar.warning("⚠️ Kon niet opslaan.")
    st.markdown("---")
    # API Keys
    st.subheader("🔑 API Keys")
    with st.expander("⚠️ Lees eerst", expanded=False):
        st.markdown("""
        Deze applicatie gebruikt uw eigen API-key voor externe providers.
        - De sleutel wordt uitsluitend gebruikt om verzoeken te sturen.
        - Er wordt niets permanent opgeslagen of gelogd.
        """)

    ack = st.checkbox("Ik wil mijn eigen API-key gebruiken.")
    if ack:
        st.session_state.gemini_api_key_user = st.text_input(
            "Gemini API Key",
            value=st.session_state.get("gemini_api_key_user", ""),
            type="password"
        )
        st.session_state.groq_api_key_user = st.text_input(
            "Groq API Key",
            value=st.session_state.get("groq_api_key_user", ""),
            type="password"
        )



# ---------------------------------------------------------
# Hoofdscherm & Chat
# ---------------------------------------------------------
st.title("Eva — Vraag & Antwoord")

# 1. Toon alle berichten uit de geschiedenis
for idx, m in enumerate(st.session_state.messages):
    if m["role"] == "image":
        st.markdown(m["content"])
        st.image(m["image_bytes"])
        st.download_button(
            label="📥 Download afbeelding",
            data=m["image_bytes"],
            file_name=f"eva_creatie_{idx}.png",
            mime="image/png",
            key=f"download_btn_{idx}"
        )
    else:
        with st.chat_message(m["role"]):
            st.write(m["content"])

# 2. Input verwerking (Tekst of Spraak)
user_input = st.chat_input("Typ je vraag…")

if not user_input and st.session_state.spoken_text_input:
    user_input = st.session_state.spoken_text_input
    st.session_state.spoken_text_input = ""

if user_input:
    # Registreer de vraag direct in de chat
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Render het bericht direct in de UI
    with st.chat_message("user"):
        st.write(user_input)

    # 1. Gebruik uitsluitend kleine letters in de triggerwoorden!
    trigger_woorden = ["genereer een afbeelding", "genereer afbeelding", "maak een afbeelding", "teken een afbeelding"]
    wil_afbeelding = any(woord in user_input.lower() for woord in trigger_woorden)

    if wil_afbeelding:
        with st.chat_message("assistant"):
            with st.spinner("Ik ben het beeld voor je aan het weven... 🎨"):
                try:
                    # Indien image.py de API-key nodig heeft, kun je eventueel
                    # st.session_state.gemini_api_key_user meesturen indien van toepassing.
                    image_path = generate_and_save_image(user_input)

                    if image_path and Path(image_path).exists():
                        with open(image_path, "rb") as f:
                            img_bytes = f.read()

                        # 2. Gebruik hier "role": "image" zodat je weergave-loop het herkent!
                        st.session_state.messages.append({
                            "role": "image",
                            "content": f"🎨 Afbeelding gegenereerd voor: *'{user_input}'*",
                            "image_bytes": img_bytes
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "Het spijt me, Frank. Het weven van de afbeelding is mislukt. Controleer of het beeldmodel goed ingesteld staat."
                        })
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Er ging iets mis tijdens het maken van de afbeelding: {e}"
                    })
    else:
        with st.chat_message("assistant"):
            with st.spinner("Aan het nadenken..."):
                try:
                    answer = answer_question(
                        user_input,
                        context="",
                        use_document_index=True
                    )
                except Exception as e:
                    answer = f"🛑 Er is een fout opgetreden: {e}"

                st.session_state.messages.append({"role": "assistant", "content": answer})

    # Ververs de pagina om geschiedenis en status netjes te synchroniseren
    st.rerun()
