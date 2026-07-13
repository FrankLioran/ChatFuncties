# web.py
import urllib.parse

def clean_duckduckgo_url(url: str) -> str:
    if "uddg=" in url:
        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        return parsed.get("uddg", [url])[0]
    return url
