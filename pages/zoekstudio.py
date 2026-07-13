import streamlit as st
import requests
import json

st.set_page_config(page_title="Eva — Zoekstudio", layout="wide")

st.title("🔎 Eva — Geavanceerde Zoekstudio")

# -----------------------------
# 1. Zoekbalk + basisinstellingen
# -----------------------------

query = st.text_input("Zoekopdracht", "quantum computing basics")

col_left, col_center, col_right = st.columns([1, 2, 1])

with col_left:
    st.markdown("### ⚙️ Instellingen")

    st.image("Eva3.jpg", caption="Eva", use_column_width=True)

    st.markdown("**Zoekmachines**")
    use_wikipedia = st.checkbox("Wikipedia", value=True)
    use_duckduckgo = st.checkbox("DuckDuckGo (web)", value=True)
    use_news = st.checkbox("News (voorbeeld)", value=False)

    st.markdown("**AI‑samenvatting**")
    use_ai_summary = st.checkbox("AI‑samenvatting inschakelen", value=True)

    if st.button("🚀 Start zoekopdracht"):
        st.session_state["zoek_query"] = query
        st.session_state["zoek_use_wikipedia"] = use_wikipedia
        st.session_state["zoek_use_duckduckgo"] = use_duckduckgo
        st.session_state["zoek_use_news"] = use_news
        st.session_state["zoek_use_ai"] = use_ai_summary

# -----------------------------
# 2. Zoeklogica
# -----------------------------

def search_wikipedia(q: str):
    """Eenvoudige Wikipedia‑zoekopdracht (samenvatting)."""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{q.replace(' ', '_')}"
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return {"engine": "Wikipedia", "ok": False, "error": f"Status {r.status_code}"}
        data = r.json()
        return {
            "engine": "Wikipedia",
            "ok": True,
            "title": data.get("title"),
            "extract": data.get("extract"),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page")
        }
    except Exception as e:
        return {"engine": "Wikipedia", "ok": False, "error": str(e)}

def search_duckduckgo(q: str):
    """DuckDuckGo Instant Answer API (geen volledige websearch, maar info/snippets)."""
    try:
        url = "https://api.duckduckgo.com"
        params = {"q": q, "format": "json", "no_redirect": 1, "no_html": 1}
        r = requests.get(url, params=params, timeout=5)
        if r.status_code != 200:
            return {"engine": "DuckDuckGo", "ok": False, "error": f"Status {r.status_code}"}
        data = r.json()
        return {
            "engine": "DuckDuckGo",
            "ok": True,
            "heading": data.get("Heading"),
            "abstract": data.get("Abstract"),
            "related_topics": data.get("RelatedTopics", [])
        }
    except Exception as e:
        return {"engine": "DuckDuckGo", "ok": False, "error": str(e)}

def search_news_example(q: str):
    """Placeholder voor een nieuws‑API (bijv. NewsAPI)."""
    # Hier zou je een echte API aanroepen met een API‑key.
    return {
        "engine": "NewsAPI (voorbeeld)",
        "ok": True,
        "articles": [
            {"title": f"Nieuws over: {q}", "source": "Demo", "url": "https://example.com"}
        ]
    }

def build_ai_summary(results: list):
    """Eenvoudige tekstuele samenvatting zonder externe AI (placeholder)."""
    # Hier kun je jouw eigen AI‑model aanroepen (Gemini, Gemma, etc.)
    # Voor nu: simpele samengestelde tekst.
    lines = []
    for r in results:
        if not r.get("ok"):
            continue
        engine = r.get("engine")
        if engine == "Wikipedia":
            lines.append(f"📚 Wikipedia zegt: {r.get('extract', '')[:300]}...")
        elif engine == "DuckDuckGo":
            lines.append(f"🦆 DuckDuckGo heading: {r.get('heading', '')}")
        elif engine.startswith("NewsAPI"):
            titles = [a["title"] for a in r.get("articles", [])]
            lines.append(f"📰 Nieuwsartikelen: " + "; ".join(titles))
    if not lines:
        return "Geen bruikbare resultaten om samen te vatten."
    return "\n\n".join(lines)

# -----------------------------
# 3. Resultaten tonen
# -----------------------------

if "zoek_query" in st.session_state and st.session_state["zoek_query"]:
    q = st.session_state["zoek_query"]

    results = []

    with col_center:
        st.markdown(f"### 🔍 Resultaten voor: `{q}`")

        if st.session_state.get("zoek_use_wikipedia"):
            res_wiki = search_wikipedia(q)
            results.append(res_wiki)
            st.markdown("#### 📚 Wikipedia")
            if res_wiki["ok"]:
                st.write(f"**Titel:** {res_wiki.get('title')}")
                st.write(res_wiki.get("extract"))
                if res_wiki.get("url"):
                    st.write(f"[Open pagina]({res_wiki['url']})")
            else:
                st.error(f"Fout bij Wikipedia: {res_wiki.get('error')}")

        if st.session_state.get("zoek_use_duckduckgo"):
            res_ddg = search_duckduckgo(q)
            results.append(res_ddg)
            st.markdown("#### 🦆 DuckDuckGo")
            if res_ddg["ok"]:
                st.write(f"**Heading:** {res_ddg.get('heading')}")
                st.write(res_ddg.get("abstract"))
                rt = res_ddg.get("related_topics", [])
                if rt:
                    st.markdown("**Gerelateerde onderwerpen:**")
                    for item in rt[:5]:
                        txt = item.get("Text") or str(item)
                        st.write(f"- {txt}")
            else:
                st.error(f"Fout bij DuckDuckGo: {res_ddg.get('error')}")

        if st.session_state.get("zoek_use_news"):
            res_news = search_news_example(q)
            results.append(res_news)
            st.markdown("#### 📰 Nieuws (voorbeeld)")
            if res_news["ok"]:
                for art in res_news.get("articles", []):
                    st.write(f"- **{art['title']}** ({art['source']}) — {art['url']}")
            else:
                st.error("Fout bij nieuwszoekopdracht.")

    with col_right:
        st.markdown("### 🧠 AI‑samenvatting")
        if st.session_state.get("zoek_use_ai"):
            summary = build_ai_summary(results)
            st.write(summary)
        else:
            st.info("AI‑samenvatting is uitgeschakeld.")
