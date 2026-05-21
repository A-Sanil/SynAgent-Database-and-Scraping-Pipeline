"""Google Patents search and document fetch via the public xhr API (no browser required)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import quote

import httpx

_log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
PATENT_ID_RE = re.compile(r"US\d{5,}[A-Z]?\d?")
XHR_QUERY = "https://patents.google.com/xhr/query"


def normalize_patent_id(publication_number: str) -> str:
    pub = publication_number.strip().upper().replace(" ", "")
    if not pub.startswith("US"):
        pub = f"US{pub}"
    return pub


def patent_url(publication_number: str) -> str:
    return f"https://patents.google.com/patent/{normalize_patent_id(publication_number)}/en"


def _http_get_json(url: str, timeout: float = 90.0, retries: int = 3) -> dict[str, Any] | None:
    """GET JSON using curl_cffi (browser TLS) when available, else httpx."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            try:
                from curl_cffi import requests as curl_requests

                response = curl_requests.get(
                    url,
                    headers=HEADERS,
                    timeout=timeout,
                    impersonate="chrome120",
                )
            except ImportError:
                response = httpx.get(url, headers=HEADERS, timeout=timeout, follow_redirects=True)

            if response.status_code in (429, 503) and attempt < retries - 1:
                import time

                time.sleep(2.0 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                import time

                time.sleep(2.0 * (attempt + 1))
                continue
    _log.warning("HTTP JSON fetch failed for %s: %s", url, last_exc)
    return None


def _xhr_get(url_path: str, timeout: float = 90.0, retries: int = 3) -> dict[str, Any] | None:
    """Call patents.google.com/xhr/query with a url= path parameter."""
    api = f"{XHR_QUERY}?url={quote(url_path, safe='')}"
    return _http_get_json(api, timeout=timeout, retries=retries)


def search_us_patents_patentsview(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Fallback: USPTO PatentsView API when Google Patents xhr is blocked."""
    api = "https://search.patentsview.org/api/v1/patent/"
    payload = {
        "q": {
            "_or": [
                {"_text_any": {"patent_title": query}},
                {"_text_any": {"patent_abstract": query}},
            ]
        },
        "f": ["patent_id", "patent_title", "patent_abstract", "patent_date"],
        "o": {"size": limit},
    }
    try:
        response = httpx.post(api, json=payload, headers=HEADERS, timeout=90.0)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        _log.warning("PatentsView search failed: %s", exc)
        return []

    patents = data.get("patents") or data.get("results") or []
    hits: list[dict[str, Any]] = []
    for item in patents[:limit]:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("patent_id") or item.get("patent_number") or "")
        if not pid:
            continue
        pub = normalize_patent_id(pid) if pid.upper().startswith("US") else f"US{pid}"
        title = item.get("patent_title") or item.get("title")
        abstract = item.get("patent_abstract") or item.get("abstract")
        hits.append(
            {
                "publication_number": pub,
                "title": _clean_html(title),
                "snippet": _clean_html(abstract),
                "assignee": None,
                "url": patent_url(pub),
                "source": "patentsview",
            }
        )
    return hits


def search_us_patents(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search US patents; returns dicts with publication_number, title, snippet, url."""
    url_path = f"q={query}&country=US&language=ENGLISH"
    data = _xhr_get(url_path)
    if not data:
        return []

    hits: list[dict[str, Any]] = []
    seen: set[str] = set()

    clusters = data.get("results", {}).get("cluster", [])
    if not isinstance(clusters, list):
        clusters = []

    for cluster in clusters:
        items = cluster.get("result", []) if isinstance(cluster, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            pub = item.get("publication_number") or item.get("patent_id")
            if not pub:
                continue
            pub = normalize_patent_id(str(pub))
            if pub in seen:
                continue
            seen.add(pub)
            hits.append(
                {
                    "publication_number": pub,
                    "title": _clean_html(item.get("title")),
                    "snippet": _clean_html(item.get("snippet")),
                    "assignee": item.get("assignee"),
                    "url": patent_url(pub),
                }
            )
            if len(hits) >= limit:
                return hits

    if len(hits) < limit:
        for pub in PATENT_ID_RE.findall(json.dumps(data)):
            pub = normalize_patent_id(pub)
            if pub in seen:
                continue
            seen.add(pub)
            hits.append({"publication_number": pub, "title": None, "snippet": None, "url": patent_url(pub)})
            if len(hits) >= limit:
                break

    if len(hits) < limit:
        for item in search_us_patents_patentsview(query, limit=limit):
            pub = item["publication_number"]
            if pub in seen:
                continue
            seen.add(pub)
            hits.append(item)
            if len(hits) >= limit:
                break

    return hits


def fetch_patent_document(publication_number: str) -> dict[str, Any]:
    """Fetch patent text metadata via xhr (works when static HTML is empty)."""
    pub = normalize_patent_id(publication_number)
    data = _xhr_get(f"patent/{pub}/en?oq=")
    if not data:
        return {"publication_number": pub, "title": None, "text": None, "url": patent_url(pub)}

    title: str | None = None
    chunks: list[str] = []

    def walk(obj: Any) -> None:
        nonlocal title
        if isinstance(obj, dict):
            t = obj.get("title")
            if isinstance(t, str) and len(t) > 3 and not title:
                title = _clean_html(t)
            for key in ("snippet", "abstract", "description", "claims", "text", "content"):
                val = obj.get(key)
                if isinstance(val, str) and len(val.strip()) > 20:
                    chunks.append(_clean_html(val))
            for val in obj.values():
                if isinstance(val, (dict, list)):
                    walk(val)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    deduped: list[str] = []
    for chunk in chunks:
        if chunk and chunk not in deduped:
            deduped.append(chunk)

    text = "\n\n".join(deduped)
    if not text:
        text = json.dumps(data, ensure_ascii=False)[:80_000]

    return {
        "publication_number": pub,
        "title": title,
        "text": text,
        "url": patent_url(pub),
        "raw_json": data,
    }


def extract_patent_urls_from_html(html: str, limit: int = 10) -> list[str]:
    """Regex/CSS fallback when xhr is unavailable."""
    if not html:
        return []
    links: list[str] = []
    seen: set[str] = set()
    for match in PATENT_ID_RE.findall(html):
        pub = normalize_patent_id(match)
        url = patent_url(pub)
        if url not in seen:
            seen.add(url)
            links.append(url)
        if len(links) >= limit:
            break
    return links


def _clean_html(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    text = text.replace("&hellip;", "...")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip() or None
