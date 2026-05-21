"""Unified Scrapling fetch helpers with static → dynamic → stealth fallback."""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import quote_plus, urljoin

_log = logging.getLogger("patent_agent.fetchers")

PATENT_PATH_RE = re.compile(r"/patent/(US[A-Z0-9]+)", re.IGNORECASE)
PATENT_FULL_RE = re.compile(
    r"https?://patents\.google\.com/patent/(US[A-Z0-9]+)",
    re.IGNORECASE,
)

SEARCH_KWARGS = {"headless": True, "network_idle": True, "timeout": 60000}
DETAIL_KWARGS = {"headless": True, "network_idle": True, "timeout": 90000}
STEALTH_KWARGS = {"headless": True, "solve_cloudflare": True, "network_idle": True, "timeout": 90000}


def _response_html(page: Any) -> str:
    for attr in ("html", "text", "body"):
        value = getattr(page, attr, None)
        if value is None:
            continue
        text = str(value)
        if len(text) > 200:
            return text
    return str(getattr(page, "text", "") or "")


def extract_patent_links(html: str, limit: int = 20) -> list[str]:
    """Extract normalized Google Patents US URLs from HTML."""
    links: list[str] = []
    seen: set[str] = set()

    def add(href: str) -> None:
        if not href:
            return
        if href.startswith("/"):
            href = urljoin("https://patents.google.com", href)
        if "/patent/US" not in href.upper():
            return
        base = href.split("?")[0].rstrip("/")
        if base not in seen:
            seen.add(base)
            links.append(base)

    for match in PATENT_FULL_RE.finditer(html):
        add(f"https://patents.google.com/patent/{match.group(1)}")
        if len(links) >= limit:
            return links

    for match in PATENT_PATH_RE.finditer(html):
        add(f"https://patents.google.com/patent/{match.group(1)}")
        if len(links) >= limit:
            return links

    return links[:limit]


def google_patents_search_url(query: str) -> str:
    return f"https://patents.google.com/?q={quote_plus(query)}&country=US"


def _fetch_static(url: str) -> Any:
    from scrapling import Fetcher

    return Fetcher.get(url)


def _fetch_dynamic(url: str, **kwargs: Any) -> Any:
    from scrapling import DynamicFetcher

    return DynamicFetcher.fetch(url, **{**DETAIL_KWARGS, **kwargs})


def _fetch_stealth(url: str, **kwargs: Any) -> Any:
    from scrapling import StealthyFetcher

    return StealthyFetcher.fetch(url, **{**STEALTH_KWARGS, **kwargs})


def fetch_page(url: str, mode: str = "auto") -> tuple[Any, str]:
    """Fetch a page. Returns (response, fetcher_used).

    Modes: static, dynamic, stealth, auto (static then dynamic then stealth).
    """
    mode = (mode or "auto").lower()
    chain: list[str]
    if mode == "auto":
        chain = ["static", "dynamic", "stealth"]
    else:
        chain = [mode]

    last_exc: Exception | None = None
    for step in chain:
        try:
            if step == "static":
                page = _fetch_static(url)
            elif step == "dynamic":
                page = _fetch_dynamic(url)
            elif step == "stealth":
                page = _fetch_stealth(url)
            else:
                continue
            html = _response_html(page)
            if len(html) > 500 or step == chain[-1]:
                return page, step
            _log.debug("Short HTML (%s chars) from %s for %s", len(html), step, url)
        except Exception as exc:
            last_exc = exc
            _log.debug("Fetcher %s failed for %s: %s", step, url, exc)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"All fetch modes failed for {url}")


def search_patent_urls(query: str, limit: int = 10, mode: str = "auto") -> tuple[list[str], str]:
    """Search Google Patents and return US patent page URLs."""
    from . import patent_search
    try:
        hits = patent_search.search_us_patents(query, limit=limit)
        links = [hit["url"] for hit in hits if hit.get("url")]
        if links:
            return links, "xhr"
    except Exception as exc:
        _log.warning("XHR search failed, falling back to browser: %s", exc)

    url = google_patents_search_url(query)
    page, used = fetch_page(url, mode=mode)
    html = _response_html(page)
    links = extract_patent_links(html, limit=limit)
    return links, used
