# pages/zoekstudio.py
import streamlit as st
import requests
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Zorg dat we modules uit de hoofdmap (zoals ai_router) kunnen importeren
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from ai_router import ask_ai

st.set_page_config(page_title="Eva — Zoekstudio", layout="wide")

st.title("🔎 Eva — Geavanceerde Zoekstudio")

# -----------------------------
# 1. Zoekbalk + basisinstellingen
# -----------------------------

query = st.text_input("Zoekopdracht", value="quantum computing basics")

col_left, col_center, col_right = st.columns([1, 2, 1])

with col_left:
    st.markdown("### ⚙️ Instellingen")

    eva_img = BASE_DIR / "images" / "Eva.jpg"

    if eva_img.exists():
        st.image(str(eva_img), caption="Eva", use_container_width=True)
    else:
        st.info("✨ Eva Zoekassistent")

with col_center: 
    st.markdown("**Zoekmachines & Bronnen**")

    use_wikipedia = st.checkbox("Wikipedia", value=True)
    use_duckduckgo = st.checkbox("DuckDuckGo (web)", value=True)
    use_arxiv = st.checkbox("ArXiv (Academisch & Scripties)", value=True)
    use_hackernews = st.checkbox("Hacker News (Tech & Discussies)", value=False)

    st.markdown("**AI‑samenvatting**")
    use_ai_summary = st.checkbox("AI‑samenvatting inschakelen", value=True)

    if st.button("🚀 Start zoekopdracht", use_container_width=True):
        st.session_state["zoek_query"] = query
        st.session_state["zoek_use_wikipedia"] = use_wikipedia
        st.session_state["zoek_use_duckduckgo"] = use_duckduckgo
        st.session_state["zoek_use_arxiv"] = use_arxiv
        st.session_state["zoek_use_hn"] = use_hackernews
        st.session_state["zoek_use_ai"] = use_ai_summary

# -----------------------------
# 2. Zoeklogica (Robuust & Modulair)
# -----------------------------

def search_wikipedia(q: str):
    """Zoekt op Wikipedia naar samenvatting."""
    try:
        headers = {"User-Agent": "EvaAssistant/1.0"}
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "opensearch", "search": q, "limit": 1, "namespace": 0, "format": "json"}

        r_search = requests.get(search_url, params=params, headers=headers, timeout=5)
        data_search = r_search.json()

        if not data_search or len(data_search) < 2 or not data_search[1]:
            return {"engine": "Wikipedia", "ok": False, "error": f"Geen Wikipedia-pagina gevonden voor '{q}'."}

        best_title = data_search[1][0]
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
    """Haalt snippets op van DuckDuckGo."""
    try:
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(q)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(url, headers=headers, timeout=5)

        if r.status_code != 200:
            return {"engine": "DuckDuckGo", "ok": False, "error": f"Status {r.status_code}"}

        raw_snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', r.text, re.DOTALL)
        clean_snippets = [re.sub(r'<[^>]+>', '', s).strip() for s in raw_snippets[:5]]

        if not clean_snippets:
            return {"engine": "DuckDuckGo", "ok": True, "snippets": ["Geen directe snippets gevonden."]}

        return {"engine": "DuckDuckGo", "ok": True, "snippets": clean_snippets}
    except Exception as e:
        return {"engine": "DuckDuckGo", "ok": False, "error": str(e)}

def search_arxiv(q: str):
    """Zoekt wetenschappelijke publicaties via de gratis ArXiv API."""
    try:
        url = f"http://export.arxiv.org/api/query?search_query=all:{requests.utils.quote(q)}&start=0&max_results=3"
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return {"engine": "ArXiv", "ok": False, "error": f"Status {r.status_code}"}

        root = ET.fromstring(r.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        articles = []

        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip().replace("\n", " ")
            summary = entry.find('atom:summary', ns).text.strip().replace("\n", " ")
            link = entry.find('atom:id', ns).text.strip()
            articles.append({"title": title, "summary": summary[:250] + "...", "url": link})

        return {"engine": "ArXiv", "ok": True, "articles": articles}
    except Exception as e:
        return {"engine": "ArXiv", "ok": False, "error": str(e)}

def search_hackernews(q: str):
    """Zoekt tech-discussies en artikelen via Algolia HN API."""
    try:
        url = f"https://hn.algolia.com/api/v1/search?query={requests.utils.quote(q)}&tags=story"
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return {"engine": "HackerNews", "ok": False, "error": f"Status {r.status_code}"}

        data = r.json()
        hits = data.get("hits", [])[:3]
        articles = []

        for item in hits:
            title = item.get("title", "Geen titel")
            url_link = item.get("url") or f"https://news.ycombinator.com/item?id={item.get('objectID')}"
            points = item.get("points", 0)
            articles.append({"title": title, "url": url_link, "points": points})

        return {"engine": "HackerNews", "ok": True, "articles": articles}
    except Exception as e:
        return {"engine": "HackerNews", "ok": False, "error": str(e)}

def generate_ai_summary(q: str, context_text: str):
    """Gebruikt de ai_router om resultaten samen te vatten."""
    messages = [
        {
            "role": "system",
            "content": "Je bent Eva, de intelligente en warme assistent van Frank. Vat de zoekresultaten helder en bondig samen in het Nederlands. Negeer eventuele irrelevante ruis of homoniemen die niet bij het hoofdthema passen."
        },
        {
            "role": "user",
            "content": f"Zoekopdracht: '{q}'\n\nGevonden informatie:\n{context_text}"
        }
    ]
    try:
        return ask_ai(messages)
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
                collected_text_for_ai.append(f"Wikipedia: {res_wiki.get('extract')}")
                if res_wiki.get("url"):
                    st.markdown(f"[🔗 Open pagina]({res_wiki['url']})")
            else:
                st.info(res_wiki.get("error"))

        # DUCKDUCKGO
        if st.session_state.get("zoek_use_duckduckgo"):
            res_ddg = search_duckduckgo(q)
            st.markdown("#### 🦆 DuckDuckGo Web")
            if res_ddg["ok"]:
                for snip in res_ddg.get("snippets", []):
                    st.write(f"- {snip}")
                    collected_text_for_ai.append(f"DuckDuckGo: {snip}")
            else:
                st.warning(f"DuckDuckGo: {res_ddg.get('error')}")

        # ARXIV
        if st.session_state.get("zoek_use_arxiv"):
            res_arxiv = search_arxiv(q)
            st.markdown("#### 🎓 ArXiv (Academisch)")
            if res_arxiv["ok"] and res_arxiv.get("articles"):
                for art in res_arxiv["articles"]:
                    st.write(f"- **[{art['title']}]({art['url']})**")
                    st.caption(art['summary'])
                    collected_text_for_ai.append(f"ArXiv Paper ({art['title']}): {art['summary']}")
            else:
                st.info("Geen academische papers gevonden.")

        # HACKER NEWS
        if st.session_state.get("zoek_use_hn"):
            res_hn = search_hackernews(q)
            st.markdown("#### 💬 Hacker News")
            if res_hn["ok"] and res_hn.get("articles"):
                for art in res_hn["articles"]:
                    st.write(f"- [{art['title']}]({art['url']}) *({art['points']} punten)*")
                    collected_text_for_ai.append(f"HackerNews: {art['title']}")
            else:
                st.info("Geen Hacker News artikelen gevonden.")

    # AI SAMENVATTING
    with col_right:
        st.markdown("### 🧠 AI‑samenvatting")
        if st.session_state.get("zoek_use_ai"):
            if collected_text_for_ai:
                with st.spinner("Eva analyseert alle bronnen..."):
                    context_full = "\n".join(collected_text_for_ai)
                    summary = generate_ai_summary(q, context_full)
                    st.info(summary)
            else:
                st.write("Geen zoekresultaten om samen te vatten.")
        else:
            st.write("AI‑samenvatting is uitgeschakeld.")
