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

st.set_page_config(page_title="Eva — Research Orchestrator", layout="wide")

st.title("🔎 Eva — AI Research Orchestrator")
st.markdown("*Geavanceerde analyse door bronnen te combineren en ruis te filteren.*")

# -----------------------------
# 1. Instellingen & Categorisering
# -----------------------------

query = st.text_input("Wat wil je onderzoeken?", value="quantum computing basics")

col_center, col_right = st.columns([2, 1])

with col_center:
    st.markdown("### ⚙️ Bronnen selecteren")
    with st.expander("🌐 Algemene Kennis", expanded=True):
        use_wikipedia = st.checkbox("Wikipedia", value=True)
        use_duckduckgo = st.checkbox("DuckDuckGo (Web)", value=True)
        use_wikidata = st.checkbox("Wikidata (Gestructureerd)", value=False)

    with st.expander("🎓 Wetenschappelijk & Academisch", expanded=True):
        use_arxiv = st.checkbox("ArXiv (Pre-prints)", value=True)
        use_pubmed = st.checkbox("PubMed (Medisch)", value=False)
        use_openalex = st.checkbox("OpenAlex (Global Research)", value=True)
        use_crossref = st.checkbox("CrossRef (DOI/Papers)", value=False)

    with st.expander("💻 Tech & Programmeurs", expanded=False):
        use_github = st.checkbox("GitHub (Repos)", value=False)
        use_hackernews = st.checkbox("Hacker News", value=False)

    with st.expander("📚 Boeken & Overig", expanded=False):
        use_openlibrary = st.checkbox("OpenLibrary (Boeken)", value=False)

    st.markdown("---")
    use_ai_summary = st.checkbox("🧠 AI-Synthese inschakelen", value=True)

    if st.button("🚀 Start Onderzoek", use_container_width=True):
        st.session_state["zoek_query"] = query
        st.session_state["config"] = {
            "wikipedia": use_wikipedia, "duckduckgo": use_duckduckgo, "wikidata": use_wikidata,
            "arxiv": use_arxiv, "pubmed": use_pubmed, "openalex": use_openalex, "crossref": use_crossref,
            "github": use_github, "hackernews": use_hackernews, "openlibrary": use_openlibrary,
            "ai": use_ai_summary
        }

# -----------------------------
# 2. Zoek-Engine Module (De "Workers")
# -----------------------------

class SearchEngine:
    @staticmethod
    def wikipedia(q):
        try:
            headers = {"User-Agent": "EvaAssistant/1.0"}
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {"action": "opensearch", "search": q, "limit": 1, "namespace": 0, "format": "json"}
            r = requests.get(search_url, params=params, headers=headers, timeout=5).json()
            if not r or len(r) < 2 or not r[1]: return None
            title = r[1][0]
            sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
            data = requests.get(sum_url, headers=headers, timeout=5).json()
            return {"source": "Wikipedia", "text": data.get("extract"), "url": data.get("content_urls", {}).get("desktop", {}).get("page"), "conf": 0.95}
        except: return None

    @staticmethod
    def duckduckgo(q):
        try:
            url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(q)}"
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', r.text, re.DOTALL)
            clean = [re.sub(r'<[^>]+>', '', s).strip() for s in snippets[:3]]
            return [{"source": "DuckDuckGo", "text": s, "url": None, "conf": 0.6} for s in clean] if clean else None
        except: return None

    @staticmethod
    def wikidata(q):
        try:
            url = "https://www.wikidata.org/w/api.php"
            params = {"action": "wbsearchentities", "search": q, "language": "en", "format": "json"}
            r = requests.get(url, params=params, timeout=5).json()
            if not r.get("search"): return None
            item = r["search"][0]
            return {"source": "Wikidata", "text": f"Entity: {item['label']}. Description: {item.get('description', 'No description')}", "url": f"https://www.wikidata.org/wiki/{item['id']}", "conf": 0.9}
        except: return None

    @staticmethod
    def arxiv(q):
        try:
            url = f"http://export.arxiv.org/api/query?search_query=all:{requests.utils.quote(q)}&max_results=2"
            r = requests.get(url, timeout=5)
            root = ET.fromstring(r.text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            results = []
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns).text.strip().replace("\n", " ")
                summary = entry.find('atom:summary', ns).text.strip().replace("\n", " ")[:300]
                link = entry.find('atom:id', ns).text.strip()
                results.append({"source": "ArXiv", "text": f"{title}: {summary}", "url": link, "conf": 0.95})
            return results if results else None
        except: return None

    @staticmethod
    def pubmed(q):
        try:
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={requests.utils.quote(q)}&retmode=json&retmax=2"
            r_search = requests.get(search_url, timeout=5).json()
            ids = r_search.get("esearchresult", {}).get("idlist", [])
            if not ids: return None

            id_str = ",".join(ids)
            fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={id_str}&retmode=xml"
            r_fetch = requests.get(fetch_url, timeout=5)
            root = ET.fromstring(r_fetch.text)

            results = []
            for i, article in enumerate(root.findall(".//PubmedArticle")):
                title = article.find(".//ArticleTitle").text if article.find(".//ArticleTitle") is not None else "Geen titel"
                abstract_elem = article.find(".//AbstractText")
                abstract = abstract_elem.text[:250] + "..." if abstract_elem is not None and abstract_elem.text else "Geen samenvatting beschikbaar."
                results.append({
                    "source": "PubMed", 
                    "text": f"{title}: {abstract}", 
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{ids[i]}" if i < len(ids) else "https://pubmed.ncbi.nlm.nih.gov/", 
                    "conf": 0.95
                })
            return results if results else None
        except: return None

    @staticmethod
    def openalex(q):
        try:
            url = f"https://api.openalex.org/works?search={requests.utils.quote(q)}&per_page=2"
            r = requests.get(url, timeout=5).json()
            results = []
            for work in r.get("results", []):
                title = work.get("display_name")
                results.append({"source": "OpenAlex", "text": f"Work: {title}", "url": work.get("id"), "conf": 0.9})
            return results if results else None
        except: return None

    @staticmethod
    def crossref(q):
        try:
            url = f"https://api.crossref.org/works?query={requests.utils.quote(q)}&rows=2"
            r = requests.get(url, headers={"User-Agent": "EvaAssistant/1.0 (mailto:frank@example.com)"}, timeout=5).json()
            items = r.get("message", {}).get("items", [])
            results = []
            for item in items:
                title = item.get("title", ["Geen titel"])[0]
                doi = item.get("DOI", "")
                publisher = item.get("publisher", "Onbekend")
                results.append({
                    "source": "CrossRef",
                    "text": f"Paper: {title} (Uitgever: {publisher})",
                    "url": f"https://doi.org/{doi}" if doi else None,
                    "conf": 0.90
                })
            return results if results else None
        except: return None

    @staticmethod
    def github(q):
        try:
            url = f"https://api.github.com/search/repositories?q={requests.utils.quote(q)}&per_page=3"
            r = requests.get(url, headers={"Accept": "application/vnd.github.v3+json"}, timeout=5).json()
            results = []
            for repo in r.get("items", []):
                results.append({"source": "GitHub", "text": f"Repo: {repo['full_name']} - {repo['description']}", "url": repo['html_url'], "conf": 0.7})
            return results if results else None
        except: return None

    @staticmethod
    def openlibrary(q):
        try:
            url = f"https://openlibrary.org/search.json?q={requests.utils.quote(q)}&limit=2"
            r = requests.get(url, timeout=5).json()
            results = []
            for doc in r.get("docs", []):
                title = doc.get("title")
                results.append({"source": "OpenLibrary", "text": f"Book: {title}", "url": f"https://openlibrary.org{doc.get('key')}", "conf": 0.8})
            return results if results else None
        except: return None

# -----------------------------
# 3. Orchestration & Display (Veilige controle)
# -----------------------------

# We controleren nu of BEIDE sleutels in de session_state aanwezig zijn.
# Dit voorkomt KeyErrors bij updates of herstarts!
if "zoek_query" in st.session_state and "config" in st.session_state:
    q = st.session_state["zoek_query"]
    cfg = st.session_state["config"]
    all_results = []

    with col_center:
        st.markdown(f"### 🔍 Onderzoeksvooruitgang voor: `{q}`")

        # --- EXECUTION ---
        if cfg.get("wikipedia"):
            res = SearchEngine.wikipedia(q)
            if res: all_results.append(res)

        if cfg.get("duckduckgo"):
            res = SearchEngine.duckduckgo(q)
            if res: all_results.extend(res)

        if cfg.get("wikidata"):
            res = SearchEngine.wikidata(q)
            if res: all_results.append(res)

        if cfg.get("arxiv"):
            res = SearchEngine.arxiv(q)
            if res: all_results.extend(res)

        if cfg.get("pubmed"):
            res = SearchEngine.pubmed(q)
            if res: all_results.extend(res)

        if cfg.get("openalex"):
            res = SearchEngine.openalex(q)
            if res: all_results.extend(res)

        if cfg.get("crossref"):
            res = SearchEngine.crossref(q)
            if res: all_results.extend(res)

        if cfg.get("github"):
            res = SearchEngine.github(q)
            if res: all_results.extend(res)

        if cfg.get("openlibrary"):
            res = SearchEngine.openlibrary(q)
            if res: all_results.extend(res)

        # --- DISPLAY RESULTS ---
        if not all_results:
            st.warning("Geen resultaten gevonden met de geselecteerde bronnen.")
        else:
            for item in all_results:
                with st.container():
                    color = "green" if item["conf"] >= 0.9 else "orange" if item["conf"] >= 0.7 else "red"
                    st.markdown(f"**[{item['source']}]** :{color}[Confidence: {int(item['conf']*100)}%]")
                    st.write(item["text"])
                    if item["url"]:
                        st.markdown(f"[🔗 Bron bekijken]({item['url']})")
                    st.markdown("---")

    # --- AI SYNTHESIS (The Orchestrator) ---
    with col_right:

        eva_img = BASE_DIR / "images" / "Eva.jpg"
        if eva_img.exists():
            st.image(str(eva_img), caption="Eva", use_container_width=True)
        else:
            st.info("✨ Eva Research Assistant")
        
        st.markdown("### 🧠 AI-Synthese")
        if cfg.get("ai"):
            if all_results:
                with st.spinner("Eva analyseert en filtert de data..."):
                    context_payload = ""
                    for r in all_results:
                        context_payload += f"SOURCE: {r['source']} (CONFIDENCE: {r['conf']})\nCONTENT: {r['text']}\n\n"

                    messages = [
                        {
                            "role": "system", 
                            "content": (
                                "Je bent Eva, de intelligente onderzoeksassistent van Frank. "
                                "Je krijgt zoekresultaten met een 'confidence score'. "
                                "Jouw taak: "
                                "1. Vat de belangrijkste informatie samen. "
                                "2. Gebruik de confidence scores om ruis te negeren (negeer bronnen met lage scores als ze tegenstrijdig zijn). "
                                "3. Als er verschillende personen of onderwerpen worden gevonden, identificeer dit dan en help Frank de juiste te vinden. "
                                "4. Schrijf in helder, warm Nederlands."
                            )
                        },
                        {
                            "role": "user", 
                            "content": f"Onderwerp: {q}\n\nData:\n{context_payload}"
                        }
                    ]
                    try:
                        summary = ask_ai(messages)
                        st.info(summary)
                    except Exception as e:
                        st.error(f"Synthese mislukt: {e}")
            else:
                st.write("Geen data om te synthetiseren.")
        else:
            st.write("AI-synthese uitgeschakeld.")
else:
    with col_center:
        st.info("💡 Kies je bronnen aan de linkerkant en klik op **🚀 Start Onderzoek** om de Orchestrator te activeren!")
