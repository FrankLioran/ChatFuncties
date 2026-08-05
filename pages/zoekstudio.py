# pages/zoekstudio.py
import streamlit as st
import requests
import re
import os
from pathlib import Path
import google.generativeai as genai

st.set_page_config(page_title="Eva — Zoekstudio", layout="wide")

st.title("🔎 Eva — Geavanceerde Zoekstudio")

# -----------------------------
# 1. Zoekbalk + basisinstellingen
# -----------------------------

query = st.text_input("Zoekopdracht", value="quantum computing basics")

col_left, col_center, col_right = st.columns([1, 2, 1])

with col_left:
    st.markdown("### ⚙️ Instellingen")

    # Pad opbouwen naar images/Eva.jpg in de hoofdmap
    BASE_DIR = Path(__file__).resolve().parent.parent
    eva_img = BASE_DIR / "images" / "Eva.jpg"

    if eva_img.exists():
        st.image(str(eva_img), caption="Eva", use_container_width=True)
    else:
        st.info("✨ Eva Zoekassistent")

with col_center: 
    st.markdown("**Zoekmachines**")

    use_wikipedia = st.checkbox("Wikipedia", value=True)
    use_duckduckgo = st.checkbox("DuckDuckGo (web)", value=True)
    use_news = st.checkbox("News (voorbeeld)", value=False)

    st.markdown("**AI‑samenvatting**")
    use_ai_summary = st.checkbox("AI‑samenvatting inschakelen (via Gemini)", value=True)

    if st.button("🚀 Start zoekopdracht", use_container_width=True):
        st.session_state["zoek_query"] = query
        st.session_state["zoek_use_wikipedia"] = use_wikipedia
        st.session_state["zoek_use_duckduckgo"] = use_duckduckgo
        st.session_state["zoek_use_news"] = use_news
        st.session_state["zoek_use_ai"] = use_ai_summary

# -----------------------------
# 2. Zoeklogica (Slim & Robuust)
# -----------------------------

def search_wikipedia(q: str):
    """Zoekt eerst naar de best passende titel en haalt daarna de samenvatting op."""
    try:
        headers = {"User-Agent": "EvaAssistant/1.0"}
        # Stap 1: Zoek op titel
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "opensearch", "search": q, "limit": 1, "namespace": 0, "format": "json"}

        r_search = requests.get(search_url, params=params, headers=headers, timeout=5)
        data_search = r_search.json()

        if not data_search or len(data_search) < 2 or not data_search[1]:
            return {"engine": "Wikipedia", "ok": False, "error": f"Geen Wikipedia-pagina gevonden voor '{q}'."}

        best_title = data_search[1][0]

        # Stap 2: Haal samenvatting op van de gevonden titel
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{best_title.replace(' ', '_')}"
        r_sum = requests.get(summary_url, headers=headers, timeout=5)

        if r_sum.status_code != 200:
            return {"engine": "Wikipedia", "ok": False, "error": f"Kon pagina '{best_title}' niet laden."}

        data = r_sum.json()
        return {
            "engine": "Wikipedia",
            "ok": True,
            "title": data.get("title", best_title),
            "extract": data.get("extract", "Geen samenvatting beschikbaar."),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page")
        }
    except Exception as e:
        return {"engine": "Wikipedia", "ok": False, "error": str(e)}

def search_duckduckgo(q: str):
    """Haalt actuele zoekresultaten/snippets op van DuckDuckGo HTML Lite."""
    try:
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(q)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(url, headers=headers, timeout=5)

        if r.status_code != 200:
            return {"engine": "DuckDuckGo", "ok": False, "error": f"Status {r.status_code}"}

        # Extract snippets met lichte regex
        raw_snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', r.text, re.DOTALL)
        clean_snippets = [re.sub(r'<[^>]+>', '', s).strip() for s in raw_snippets[:5]]

        if not clean_snippets:
            return {"engine": "DuckDuckGo", "ok": True, "snippets": ["Geen directe snippets gevonden."]}

        return {
            "engine": "DuckDuckGo",
            "ok": True,
            "snippets": clean_snippets
        }
    except Exception as e:
        return {"engine": "DuckDuckGo", "ok": False, "error": str(e)}

def search_news_example(q: str):
    return {
        "engine": "NewsAPI (voorbeeld)",
        "ok": True,
        "articles": [
            {"title": f"Recent nieuws & ontwikkelingen rondom: {q}", "source": "Tech News Daily"}
        ]
    }

def generate_ai_summary_with_gemini(q: str, context_text: str):
    """Gebruikt Gemini om een échte, slimme samenvatting te schrijven."""
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ Gemini API key niet ingesteld. Kan geen AI-samenvatting genereren."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
        Je bent Eva, de intelligente en warme assistent van Frank.
        Vat de onderstaande zoekresultaten voor de zoekopdracht '{q}' helder, overzichtelijk en vlot samen in het Nederlands.

        Zoekresultaten:
        {context_text}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Kon geen AI-samenvatting maken: {str(e)}"

# -----------------------------
# 3. Resultaten tonen
# -----------------------------

if "zoek_query" in st.session_state and st.session_state["zoek_query"]:
    q = st.session_state["zoek_query"]
    collected_text_for_ai = []

    with col_center:
        st.markdown(f"### 🔍 Resultaten voor: `{q}`")

        # WIKIPEDIA
        if st.session_state.get("zoek_use_wikipedia"):
            res_wiki = search_wikipedia(q)
            st.markdown("#### 📚 Wikipedia")
            if res_wiki["ok"]:
                st.write(f"**{res_wiki.get('title')}**")
                st.write(res_wiki.get("extract"))
                collected_text_for_ai.append(f"Wikipedia ({res_wiki.get('title')}): {res_wiki.get('extract')}")
                if res_wiki.get("url"):
                    st.markdown(f"[🔗 Open Wikipedia pagina]({res_wiki['url']})")
            else:
                st.warning(res_wiki.get("error"))

        # DUCKDUCKGO
        if st.session_state.get("zoek_use_duckduckgo"):
            res_ddg = search_duckduckgo(q)
            st.markdown("#### 🦆 DuckDuckGo Web-resultaten")
            if res_ddg["ok"]:
                for snip in res_ddg.get("snippets", []):
                    st.write(f"- {snip}")
                    collected_text_for_ai.append(f"DuckDuckGo: {snip}")
            else:
                st.error(f"Fout bij DuckDuckGo: {res_ddg.get('error')}")

        # NIEUWS
        if st.session_state.get("zoek_use_news"):
            res_news = search_news_example(q)
            st.markdown("#### 📰 Nieuws")
            if res_news["ok"]:
                for art in res_news.get("articles", []):
                    st.write(f"- **{art['title']}** ({art['source']})")
                    collected_text_for_ai.append(f"Nieuws: {art['title']}")

    # AI SAMENVATTING
    with col_right:
        st.markdown("### 🧠 AI‑samenvatting")
        if st.session_state.get("zoek_use_ai"):
            if collected_text_for_ai:
                with st.spinner("Eva analyseert de zoekresultaten..."):
                    context_full = "\n".join(collected_text_for_ai)
                    summary = generate_ai_summary_with_gemini(q, context_full)
                    st.info(summary)
            else:
                st.write("Geen zoekresultaten om samen te vatten.")
        else:
            st.write("AI‑samenvatting is uitgeschakeld.")
