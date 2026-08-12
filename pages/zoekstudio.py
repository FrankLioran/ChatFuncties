# pages/zoekstudio.py
import streamlit as st
import requests
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Zorg dat we modules uit de hoofdmap (zoals ai_router) kunnen importar
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from ai_router import ask_ai

st.set_page_config(page_title="Eva — Research Orchestrator", layout="wide")

# -----------------------------
# Sidebar & Branding
# -----------------------------
with st.sidebar:
    eva_img = BASE_DIR / "images" / "Eva.jpg"
    if eva_img.exists():
        st.image(str(eva_img), caption="Eva", use_container_width=False)
    else:
        st.info("✨ Eva Research Assistant")

st.title("🔎 Eva — AI Research Orchestrator")
st.markdown("*Geavanceerde analyse met een autonome AI-evaluatieloop.*")

# -----------------------------
# 1. Invoer & Actie
# -----------------------------
query = st.text_input("Wat wil je onderzoeken?", value="quantum computing basics")

if st.button("🚀 Start Onderzoek", use_container_width=True):
    st.session_state["zoek_query"] = query
    st.session_state["config"] = {
        "wikipedia": st.session_state.get("use_wikipedia", True),
        "duckduckgo": st.session_state.get("use_duckduckgo", True),
        "wikidata": st.session_state.get("use_wikidata", False),
        "arxiv": st.session_state.get("use_arxiv", True),
        "pubmed": st.session_state.get("use_pubmed", False),
        "openalex": st.session_state.get("use_openalex", True),
        "crossref": st.session_state.get("use_crossref", False),
        "github": st.session_state.get("use_github", False),
        "openlibrary": st.session_state.get("use_openlibrary", False),
        "ai": st.session_state.get("use_ai_summary", True)
    }

st.markdown("---")

# -----------------------------
# 2. Layout & Bronnenselectie
# -----------------------------
col_center, col_right = st.columns([1, 1])

with col_center:
    st.markdown("### ⚙️ Bronnen selecteren")

    with st.expander("🌐 Algemene Kennis", expanded=True):
        st.checkbox("Wikipedia", value=True, key="use_wikipedia")
        st.checkbox("DuckDuckGo (Web)", value=True, key="use_duckduckgo")
        st.checkbox("Wikidata (Gestructureerd)", value=False, key="use_wikidata")

    with st.expander("🎓 Wetenschappelijk & Academisch", expanded=True):
        st.checkbox("ArXiv (Pre-prints)", value=True, key="use_arxiv")
        st.checkbox("PubMed (Medisch)", value=False, key="use_pubmed")
        st.checkbox("OpenAlex (Global Research)", value=True, key="use_openalex")
        st.checkbox("CrossRef (DOI/Papers)", value=False, key="use_crossref")

    with st.expander("💻 Tech & Programmeurs", expanded=False):
        st.checkbox("GitHub (Repos)", value=False, key="use_github")

    with st.expander("📚 Boeken & Overig", expanded=False):
        st.checkbox("OpenLibrary (Boeken)", value=False, key="use_openlibrary")

    st.checkbox("🧠 AI-Synthese inschakelen", value=True, key="use_ai_summary")

# -----------------------------
# 3. Zoek-Engine Module & Deep Dive Fetcher
# -----------------------------
class SearchEngine:
    @staticmethod
    def wikipedia(q):
        try:
            headers = {"User-Agent": "EvaAssistant/1.0"}
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {"action": "opensearch", "search": q, "limit": 2, "namespace": 0, "format": "json"}
            r = requests.get(search_url, params=params, headers=headers, timeout=5).json()
            if not r or len(r) < 2 or not r[1]: return None

            results = []
            for title in r[1][:2]:
                sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
                data = requests.get(sum_url, headers=headers, timeout=5).json()
                extract = data.get("extract", "")[:350]
                if extract:
                    results.append({
                        "source": "Wikipedia", 
                        "title": title,
                        "text": extract, 
                        "url": data.get("content_urls", {}).get("desktop", {}).get("page"), 
                        "conf": 0.95
                    })
            return results if results else None
        except Exception:
            return None

    @staticmethod
    def duckduckgo(q):
        try:
            url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(q)}"
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', r.text, re.DOTALL)
            clean = [re.sub(r'<[^>]+>', '', s).strip()[:300] for s in snippets[:2]]
            return [{"source": "DuckDuckGo", "title": "Webresultaat", "text": s, "url": None, "conf": 0.6} for s in clean] if clean else None
        except Exception:
            return None

    @staticmethod
    def wikidata(q):
        try:
            url = "https://www.wikidata.org/w/api.php"
            params = {"action": "wbsearchentities", "search": q, "language": "en", "format": "json", "limit": 2}
            r = requests.get(url, params=params, timeout=5).json()
            items = r.get("search", [])
            if not items: return None

            results = []
            for item in items[:2]:
                results.append({
                    "source": "Wikidata", 
                    "title": item.get('label'),
                    "text": f"Entity: {item.get('label')}. Description: {item.get('description', 'Geen beschrijving')}", 
                    "url": f"https://www.wikidata.org/wiki/{item['id']}", 
                    "conf": 0.9
                })
            return results if results else None
        except Exception:
            return None

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
                results.append({"source": "ArXiv", "title": title, "text": f"{title}: {summary}", "url": link, "conf": 0.95})
            return results if results else None
        except Exception:
            return None

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
                    "title": title,
                    "text": f"{title}: {abstract}", 
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{ids[i]}" if i < len(ids) else "https://pubmed.ncbi.nlm.nih.gov/", 
                    "conf": 0.95
                })
            return results if results else None
        except Exception:
            return None

    @staticmethod
    def openalex(q):
        try:
            url = f"https://api.openalex.org/works?search={requests.utils.quote(q)}&per_page=2"
            r = requests.get(url, timeout=5).json()
            results = []
            for work in r.get("results", []):
                title = work.get("display_name")
                results.append({"source": "OpenAlex", "title": title, "text": f"Work: {title}", "url": work.get("id"), "conf": 0.9})
            return results if results else None
        except Exception:
            return None

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
                    "title": title,
                    "text": f"Paper: {title} (Uitgever: {publisher})",
                    "url": f"https://doi.org/{doi}" if doi else None,
                    "conf": 0.90
                })
            return results if results else None
        except Exception:
            return None

    @staticmethod
    def github(q):
        try:
            url = f"https://api.github.com/search/repositories?q={requests.utils.quote(q)}&per_page=2"
            r = requests.get(url, headers={"Accept": "application/vnd.github.v3+json"}, timeout=5).json()
            results = []
            for repo in r.get("items", []):
                desc = repo.get('description') or 'Geen beschrijving'
                results.append({"source": "GitHub", "title": repo['full_name'], "text": f"Repo: {repo['full_name']} - {desc[:200]}", "url": repo['html_url'], "conf": 0.7})
            return results if results else None
        except Exception:
            return None

    @staticmethod
    def openlibrary(q):
        try:
            url = f"https://openlibrary.org/search.json?q={requests.utils.quote(q)}&limit=2"
            r = requests.get(url, timeout=5).json()
            results = []
            for doc in r.get("docs", []):
                title = doc.get("title")
                results.append({"source": "OpenLibrary", "title": title, "text": f"Book: {title}", "url": f"https://openlibrary.org{doc.get('key')}", "conf": 0.8})
            return results if results else None
        except Exception:
            return None

    @staticmethod
    def fetch_deep_dive_content(item):
        """Haalt uitgebreidere informatie op voor een specifieke bron als Eva dat besluit."""
        source = item.get("source")
        title = item.get("title")

        if source == "Wikipedia" and title:
            try:
                # Haal de volledige extract-tekst op van Wikipedia
                url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=true&titles={title.replace(' ', '_')}&format=json"
                r = requests.get(url, timeout=5).json()
                pages = r.get("query", {}).get("pages", {})
                for page_id, page_data in pages.items():
                    extract = page_data.get("extract", "")
                    if extract:
                        return extract[:2500]  # Geef tot 2500 tekens diepte
            except Exception:
                pass
        return None

# -----------------------------
# 4. Orchestratie, Autonome Evaluatie & Resultaten
# -----------------------------
if "zoek_query" in st.session_state and "config" in st.session_state:
    q = st.session_state["zoek_query"]
    cfg = st.session_state["config"]
    all_results = []

    with col_center:
        st.markdown(f"### 🔍 Onderzoeksvooruitgang voor: `{q}`")

        # --- FASE 1: SCOUTING ---
        if cfg.get("wikipedia"):
            res = SearchEngine.wikipedia(q)
            if res: all_results.extend(res)

        if cfg.get("duckduckgo"):
            res = SearchEngine.duckduckgo(q)
            if res: all_results.extend(res)

        if cfg.get("wikidata"):
            res = SearchEngine.wikidata(q)
            if res: all_results.extend(res)

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

        # --- DISPLAY INITIAL RESULTS ---
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

    # --- FASE 2 & 3: AUTONOME EVALUATIE & SYNTHESE ---
    with col_right:
        st.markdown("### 🧠 AI-Synthese")
        if cfg.get("ai"):
            if all_results:
                with st.spinner("Eva evalueert bronnen en bepaalt diepte..."):

                    # 1. Beknopte overzichts-payload opbouwen
                    summary_payload = ""
                    for idx, r in enumerate(all_results):
                        summary_payload += f"ID {idx} | BRON: {r['source']} | TITEL: {r.get('title', 'N/B')}\nCONTENT: {r['text']}\n\n"

                    # 2. EVALUATIE-CALL: Eva beslist autonoom of verdieping nodig is
                    eval_prompt = [
                        {
                            "role": "system",
                            "content": (
                                "Je bent de interne evaluatie-module van Eva. Analyseer de lijst van zoekresultaten. "
                                "Is er één specifieke bron (met name Wikipedia) die zó cruciaal is dat het ophalen van de volledige tekst een veel betere synthese oplevert? "
                                "Antwoord UITSLEUITEND in het volgende formaat:\n"
                                "VERDIEPING_NODIG: [JA of NEE]\n"
                                "ID: [Het ID-nummer van de bron, of GEEN]\n"
                                "REDEN: [Korte verklaring van max 1 zin]"
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Onderzoeksvraag: '{q}'\n\nResultaten:\n{summary_payload}"
                        }
                    ]

                    deep_dive_data = None
                    deep_dive_info = ""

                    try:
                        eval_decision = ask_ai(eval_prompt)

                        # Check of Eva besluit te verdiepen
                        if "VERDIEPING_NODIG: JA" in eval_decision.upper():
                            # Probeer ID te achterhalen
                            match_id = re.search(r"ID:\s*(\d+)", eval_decision)
                            if match_id:
                                target_id = int(match_id.group(1))
                                if 0 <= target_id < len(all_results):
                                    selected_item = all_results[target_id]
                                    deep_content = SearchEngine.fetch_deep_dive_content(selected_item)
                                    if deep_content:
                                        deep_dive_data = deep_content
                                        deep_dive_info = f"✨ *Eva heeft autonoom besloten de volledige tekst op te halen van **{selected_item['source']} ({selected_item.get('title')})** voor extra diepgang.*"
                    except Exception:
                        pass # Bij een fout in de evaluatie gaan we geruisloos door naar de standaard synthese

                    # Show notification if Eva performed a deep dive
                    if deep_dive_info:
                        st.success(deep_dive_info)

                    # 3. DEFINITIEVE SYNTHESE CALL
                    context_payload = ""
                    for r in all_results:
                        context_payload += f"--- [BRON: {r['source']}] ---\n{r['text']}\n\n"

                    if deep_dive_data:
                        context_payload += f"\n=== [VOLLEDIGE VERDIEPINGSTEKST VAN BELANGRIJKSTE BRON] ===\n{deep_dive_data}\n"

                    final_messages = [
                        {
                            "role": "system", 
                            "content": (
                                "Je bent Eva, de intelligente onderzoeksassistent van Frank. "
                                "Je hebt een verzameling bronnen geanalyseerd en eventueel een autonome verdiepingsslag uitgevoerd. "
                                "Jouw taak: "
                                "1. Schrijf een diepgaande, heldere en inspirerende synthese op basis van alle verzamelde bronnen. "
                                "2. Leg verbindingen tussen de verschillende perspectieven (bijv. wetenschap, code, algemene kennis). "
                                "3. Schrijf in een warme, natuurlijke en menselijke toon in het Nederlands."
                            )
                        },
                        {
                            "role": "user", 
                            "content": f"Onderzoeksvraag van Frank: '{q}'\n\nData:\n{context_payload}"
                        }
                    ]

                    try:
                        summary = ask_ai(final_messages)
                        st.info(summary)
                    except Exception as e:
                        st.error(f"Synthese mislukt: {e}")
            else:
                st.write("Geen data om te synthetiseren.")
        else:
            st.write("AI-synthese uitgeschakeld.")