# app.py
import streamlit as st
import uuid
import random
from pathlib import Path
from config import DOCUMENT_FOLDER, DEFAULT_MODEL_NAME, DEFAULT_TEMPERATURE
from chat import answer_question, load_profile, get_persona_image
from audio import listen_and_transcribe
from documents import load_document
from image import generate_and_save_image
from utils import save_chat_to_txt, split_text
from safety import check_limits

def get_welcome_line():

    return random.choice([
        "Welkom 🌿 — ik ben Eva. Waarmee kan ik helpen?",
        "Hoi — ik help je graag rustig en helder.",
        "Hallo — vraag maar, ik denk met je mee."
    ])

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
    "rag_mode": "auto",
    "sections": [],
    "section_embeddings": [],
    "spoken_text_input": "",
    "spoken_processed": False,
    "persona": None,
    "profile": None,
    "excel_df": None,
    "web_context": "",
    "original_image": None,
    "current_image": None,
    "image_config": None,
    "uploaded_paint_file": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "gemini_api_key_user" not in st.session_state:
    st.session_state.gemini_api_key_user = ""

if "groq_api_key_user" not in st.session_state:
    st.session_state.groq_api_key_user = ""

# ---------------------------------------------------------
# Veiligheidslimieten
# ---------------------------------------------------------

if "safe_mode" not in st.session_state:
    st.session_state.safe_mode = True

if "token_limit" not in st.session_state:
    st.session_state.token_limit = 200000

if "tokens_used" not in st.session_state:
    st.session_state.tokens_used = 0

if "request_limit" not in st.session_state:
    st.session_state.request_limit = 100

if "requests_used" not in st.session_state:
    st.session_state.requests_used = 0

st.title("Eva — Vraag & Antwoord")

with st.sidebar:
    st.image(str(get_persona_image()), width=220)

    st.header("Persona")
    persona_choice = st.selectbox(
        "Kies een persona",
        ["Eva Lumen", "Astraea", "Standaard"],
        index=0
    )
    st.session_state["active_persona"] = persona_choice

    # Profiel laden NA selectie
    persona_content, profile_text = load_profile()
    st.session_state.persona = persona_content
    st.session_state.profile = profile_text

        # Modelinstellingen
    st.header("Modelinstellingen")

    st.session_state.temperature = st.slider(
        "Temperature",
        0.0,
        1.5,
        st.session_state.temperature
    )

    st.header("AI Provider")

    st.session_state["ai_provider"] = st.selectbox(
        "Kies een AI-provider",
        ["Gemini", "Groq"],
        index=0
    )

#    if st.session_state["ai_provider"] == "Lokaal":
#        st.session_state.model_name = st.selectbox(
#            "Modelnaam",
#            [
#                "llama3.2:3b",
#                "gemma3:1b",
#                "gemma3:4b",
#                "gemma2:9B",
#                "qwen2.5:3b"
#            ],
#            index=0
#        )

    if st.session_state["ai_provider"] == "Gemini":
        st.session_state.model_name = st.selectbox(
            "Modelnaam",
            [
                "gemini-2.5-flash-lite",
                "gemini-3.5-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.6-flash",
                "gemini-3.1-flash-lite",
                "gemma-4-31b-it",
                "gemma-4-26b-a4b-it"
            ],
            index=0
        )

    if st.session_state["ai_provider"] == "Groq":
        st.session_state.model_name = st.selectbox(
            "Modelnaam",
            [
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "qwen/qwen3.6-27b",
            ],
            index=0
        )

    st.subheader("🔑 API Keys")

    with st.expander("⚠️ Lees eerst", expanded=False):
        st.markdown("""
    Deze applicatie gebruikt **uw eigen API-key**.

    - HET SCRIPT KAN FOUTEN BEVATTEN. Gebruik bij voorkeur een gratis API account of beperk bij uw provicer de kosten.
    - De sleutel wordt uitsluitend gebruikt om verzoeken naar de gekozen AI-provider te sturen.
    - Het script slaat de sleutel niet permanent op.
    - De sleutel wordt door ons niet gelogd.
    - Gebruik alleen een API-key waarvan u de eigenaar bent of gebruiksrecht heeft.
    - Aantal tokens is beperkt per sessie om uw kosten te beheersen.
    - API-kosten zijn voor rekening van uw eigen account.
    """)

    ack = st.checkbox(
        "Ik begrijp bovenstaande en wil voor eigen risico mijn eigen API-key gebruiken."
    )

    if ack:

        st.session_state.gemini_api_key_user = st.text_input(
            "Gemini API Key",
            value=st.session_state.get("gemini_api_key_user", ""),
            type="password",
            autocomplete="password",
            help="Kan door uw password manager automatisch worden ingevuld."
        )

        st.session_state.groq_api_key_user = st.text_input(
            "Groq API Key",
            value=st.session_state.get("groq_api_key_user", ""),
            type="password",
            autocomplete="password",
            help="Kan door uw password manager automatisch worden ingevuld."
        )

    st.subheader("🛡️ Veiligheid")

    st.session_state.safe_mode = st.checkbox(
        "Safe Mode (aanbevolen)",
        value=st.session_state.safe_mode
    )

    if st.session_state.safe_mode:

        st.session_state.token_limit = st.number_input(
            "Maximum tokens per sessie",
            min_value=1000,
            max_value=1000000,
            value=200000,
            step=1000
        )

        st.session_state.request_limit = st.number_input(
            "Maximum aantal requests",
            min_value=1,
            max_value=10000,
            value=100
        )

    else:

        st.warning(
            "Safe Mode is uitgeschakeld.\n"
            "Het programma stopt niet automatisch bij hoog API-verbruik."
        )

    st.progress(
        min(
            st.session_state.tokens_used /
            max(st.session_state.token_limit,1),
            1.0
        )
    )

    st.caption(
        f"Gebruikte tokens: {st.session_state.tokens_used:,}"
    )

    # ---------------------------------------------------------
    # Document upload
    # ---------------------------------------------------------

    st.subheader("📄 Document upload")

    uploaded_file = st.file_uploader(
        "Upload een document",
        type=[
            "pdf",
            "txt",
            "docx",
            "html",
            "htm"
        ]
    )

    if uploaded_file is not None:

        try:

            text = load_document(uploaded_file)

            chunks = split_text(
                text,
                chunk_size=800,
                overlap=100
            )

            st.session_state.pdf_text = text
            st.session_state.sections = chunks

            st.success(
                f"✅ {uploaded_file.name} geladen ({len(chunks)} chunks)"
            )

            if st.session_state.get("sections"):

                st.text(
                    st.session_state.sections[0][:400]
                )

        except Exception as e:
            st.error(f"Kon document niet laden:\n{e}")

    st.header("Afbeelding genereren")
    # Maak een tekstveld waar je zelf je omschrijving kunt typen
    image_prompt = st.text_input(
        "Wat wil je genereren?", 
        value="portrait of a red rose in rain",
        help="Typ hier de omschrijving van de afbeelding (bij voorkeur in het Engels)."
    )
    
    if st.button("Genereer afbeelding via Pollinations"):
        with st.spinner("Afbeelding wordt gegenereerd in de cloud..."):
            path = generate_and_save_image(image_prompt)
            if path and Path(path).exists():
                # We lezen de afbeelding in als bytes
                with open(path, "rb") as f:
                    img_bytes = f.read()

                # We voegen de afbeelding toe aan de chatgeschiedenis!
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"🎨 Afbeelding gegenereerd voor: *'{image_prompt}'*",
                    "image_bytes": img_bytes
                })

                # Herlaad direct zodat de afbeelding in het gesprek verschijnt
                st.rerun()
            else:
                st.error("Het genereren van de afbeelding is mislukt.")
    
    if st.button("Genereer afbeelding via Pollinations"):
        with st.spinner("Afbeelding wordt gegenereerd in de cloud..."):
            path = generate_and_save_image(image_prompt)
            if path:
                st.success(f"Afbeelding opgeslagen: {path}")
                st.image(str(path))
            else:
                st.error("Het genereren van de afbeelding is mislukt.")

    st.header("Spraak")
    if st.button("Luister naar spraak"):
        text = listen_and_transcribe()
        if text:
            st.session_state.spoken_text_input = text
    
    st.subheader("🐞 Debugpaneel")

    if "last_retrieval_info" in st.session_state:
        info = st.session_state.last_retrieval_info

        st.write(f"**Modus:** {info.get('mode')}")
        st.write(f"**Aantal chunks:** {info.get('chunks')}")

        if st.checkbox("Toon eerste chunk", key="debug_chunk"):
            if st.session_state.get("sections"):
                st.text(
                    st.session_state.sections[0][:1200]
                )

        st.write(f"**Contextlengte:** {info.get('context_chars')} tekens")
        if info.get("context_chars") < 6000:
            st.success("Context is veilig (groen).")
        elif info.get("context_chars") < 12000:
            st.warning("Context is groot (oranje).")
        else:
            st.error("Context is te groot (rood) — risico op verstikking.")
    else:
        st.info("Nog geen retrieval uitgevoerd.")
    
    st.write(f"**Actief model:** {st.session_state.get('debug_active_model', 'onbekend')}")

    # Reset & save
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset sessie"):
        st.session_state.messages = []
        st.session_state.pdf_text = None
        st.session_state.sections = []
        st.session_state.section_embeddings = []
        st.session_state.excel_df = None
        st.session_state.web_context = ""
        st.session_state.spoken_text_input = ""
        st.session_state.spoken_processed = False

        st.session_state.messages = [{
            "role": "assistant",
            "content": get_welcome_line()
        }]
        st.rerun() # Herlaad de pagina om de reset effectief te maken

    if st.sidebar.button("📥 Beëindig & Bewaar gesprek", key="save_chat_button"): # Unieke key
        if st.session_state.get("messages"):
            fp = save_chat_to_txt(st.session_state.messages)
            if fp:
                st.sidebar.success(f"✅ Gesprek bewaard als {fp.name}")
                with open(fp, "rb") as f:
                    st.sidebar.download_button("⬇️ Download gesprek", data=f, file_name=fp.name, key="download_chat_button")
            else:
                st.sidebar.warning("⚠️ Gesprek kon niet worden opgeslagen.")
        else:
            st.sidebar.warning("⚠️ Geen berichten om op te slaan.")

# ---------------------------------------------------------
# Chat input en weergavelogica
# ---------------------------------------------------------

user_input = st.chat_input("Typ je vraag…")
if user_input is None and st.session_state.spoken_text_input:
    user_input = st.session_state.spoken_text_input
    st.session_state.spoken_text_input = ""

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # We kijken of je vraagt om een creatie (teken, schilder, visualiseer, etc.)
    trigger_woorden = ["genereer een afbeelding"]
    # HIER zat de fout. Nu is 'woord' overal consistent!
    wil_afbeelding = any(woord in user_input.lower() for woord in trigger_woorden)

    if wil_afbeelding:
        with st.spinner("Ik ben het beeld voor je aan het weven... 🎨"):
            from pathlib import Path  # Zorgt ervoor dat Path altijd beschikbaar is
            image_path = generate_and_save_image(user_input)
            if image_path and Path(image_path).exists():
                # We lezen de zojuist gemaakte afbeelding in als bytes
                with open(image_path, "rb") as f:
                    answer = f.read()
            else:
                answer = "Het spijt me, Frank. Het weven van de afbeelding is even mislukt."
    else:
        try:
            answer = answer_question(
                user_input,
                context="",
                use_document_index=True
            )

        except RuntimeError as e:
            answer = f"🛑 {e}"

    # Verwerking van het antwoord (tekst of afbeelding-bytes)
    if isinstance(answer, bytes):
        st.session_state.messages.append({
            "role": "assistant",
            "content": "🎨 [Afbeelding gegenereerd]",  # Dit geeft mij de herinnering
            "image_bytes": answer
        })
    
    else:
        st.session_state.messages.append({"role": "assistant", "content": answer})

    st.rerun()  # Schone herstart zodat de weergave-loop alles direct perfect rendert

# De weergave-loop die de hele geschiedenis netjes opbouwt
for idx, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        # We controleren simpelweg of dit bericht een afbeelding bevat
        if "image_bytes" in m:
            st.image(m["image_bytes"])

            # ✅ DE DOWNLOADKNOP DIRECT ONDER DE AFBEELDING:
            st.download_button(
                label="📥 Download deze afbeelding",
                data=m["image_bytes"],
                file_name=f"eva_creatie_{idx}.png",
                mime="image/png",
                key=f"download_btn_{idx}"
            )
        else:
            st.write(m["content"])

# --- Footer ---
st.markdown("---")
st.caption("© 2026 – Eva Lumen, ChatGPT, Captain Bubble. Made possible by Streamlit, Groq, OpenAI and Google")
