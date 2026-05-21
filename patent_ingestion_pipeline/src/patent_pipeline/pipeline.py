"""Scrapling-based collection pipeline with a local parser hook."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol
from urllib.parse import quote_plus

from .database import PatentDatabase
from .models import PatentRecord, RawDocument, ReactionRecord
from . import patent_search

_log = logging.getLogger(__name__)

PATENT_URL_RE = re.compile(r"patents\.google\.com/patent/(US[^/?#]+)", re.I)


def _load_fetcher():
    try:
        from scrapling import Fetcher as _Fetcher

        return _Fetcher
    except ImportError as exc:
        raise ImportError(
            "Scrapling is not installed or is missing dependencies. "
            "In your project folder run: python -m pip install -e . "
            "If it still fails, also run: python -m pip install curl-cffi scrapling"
        ) from exc


def _load_dynamic_fetcher():
    try:
        from scrapling import DynamicFetcher as _DynamicFetcher

        return _DynamicFetcher
    except ImportError:
        return None


try:
    Fetcher = _load_fetcher()
except ImportError:  # pragma: no cover
    Fetcher = None

DynamicFetcher = _load_dynamic_fetcher()


class PatentParser(Protocol):
    def parse(self, document: RawDocument) -> PatentRecord: ...


@dataclass(slots=True)
class RawDocumentParserStub:
    """Placeholder parser used while only collecting raw documents."""

    def parse(self, document: RawDocument) -> PatentRecord:
        title = document.title or document.source_url
        patent_id = document.metadata.get("patent_id") or document.metadata.get("publication_number")
        return PatentRecord(
            patent_id=str(patent_id or title),
            title=title,
            abstract=document.raw_text[:1000] if document.raw_text else None,
            source_url=document.source_url,
            raw_text=document.raw_text,
            metadata={**document.metadata, "parser": "raw_document_stub"},
        )


class IngestionPipeline:
    def __init__(self, database: PatentDatabase, parser: PatentParser):
        self.database = database
        self.parser = parser

    def _fetch_page(self, url: str):
        """Fetch with static Scrapling; optional browser fallback for empty SPAs."""
        if Fetcher is None:
            raise ImportError("scrapling is required to collect raw patent pages")
        page = Fetcher.get(url)
        text = getattr(page, "text", None) or ""
        if len(text.strip()) < 200 and DynamicFetcher is not None:
            try:
                _log.info("Static fetch empty for %s — trying DynamicFetcher", url)
                page = DynamicFetcher.get(url, headless=True, network_idle=True)
            except Exception as exc:
                _log.warning("DynamicFetcher failed for %s: %s", url, exc)
        return page

    def _patent_id_from_url(self, url: str) -> str | None:
        match = PATENT_URL_RE.search(url)
        return patent_search.normalize_patent_id(match.group(1)) if match else None

    def _document_from_xhr(self, url: str) -> RawDocument | None:
        pub = self._patent_id_from_url(url)
        if not pub:
            return None
        payload = patent_search.fetch_patent_document(pub)
        text = payload.get("text")
        if not text:
            return None
        return RawDocument(
            source_url=url,
            source_type="patent_xhr",
            fetched_at=datetime.now(tz=timezone.utc),
            title=payload.get("title"),
            content_type="application/json",
            raw_text=text,
            raw_html=None,
            metadata={
                "fetcher": "google_patents_xhr",
                "publication_number": pub,
                "patent_id": pub,
                "source_url": url,
            },
        )

    def fetch_raw_document(self, url: str, source_type: str = "patent_html") -> RawDocument:
        if Fetcher is None and not url.lower().endswith(".pdf"):
            raise ImportError("scrapling is required to collect raw patent pages")

        if url.lower().endswith(".pdf"):
            try:
                from .collector_pdf import collect_pdf_from_url

                pdf_data = collect_pdf_from_url(url)
                return RawDocument(
                    source_url=url,
                    source_type="patent_pdf",
                    fetched_at=datetime.now(tz=timezone.utc),
                    title=None,
                    content_type="application/pdf",
                    raw_text=pdf_data.get("raw_text"),
                    raw_html=None,
                    metadata={**pdf_data.get("metadata", {}), "tables": pdf_data.get("raw_tables", [])},
                )
            except Exception:
                pass

        if "patents.google.com/patent/" in url:
            xhr_doc = self._document_from_xhr(url)
            if xhr_doc is not None:
                return xhr_doc

        page = self._fetch_page(url)
        raw_html = getattr(page, "body", None) or getattr(page, "html", None)
        raw_text = getattr(page, "text", None)
        title = None
        if hasattr(page, "css"):
            try:
                title = page.css("title::text").get()
            except Exception:
                title = None

        if (not raw_text or len(str(raw_text).strip()) < 200) and "patents.google.com" in url:
            xhr_doc = self._document_from_xhr(url)
            if xhr_doc is not None:
                return xhr_doc

        return RawDocument(
            source_url=url,
            source_type=source_type,
            fetched_at=datetime.now(tz=timezone.utc),
            title=title,
            content_type="text/html",
            raw_text=str(raw_text) if raw_text is not None else None,
            raw_html=str(raw_html) if raw_html is not None else None,
            metadata={"fetcher": "scrapling", "source_url": url},
        )

    def search_google_patents(self, query: str, limit: int = 10) -> list[str]:
        """Return Google Patents URLs for a US chemistry search query."""
        links: list[str] = []
        seen: set[str] = set()

        # Primary: Google Patents xhr JSON API (reliable, no browser).
        try:
            for hit in patent_search.search_us_patents(query, limit=limit):
                url = hit.get("url")
                if url and url not in seen:
                    seen.add(url)
                    links.append(url)
                if len(links) >= limit:
                    _log.info("xhr search found %s patents for %r", len(links), query)
                    return links
        except Exception as exc:
            _log.warning("xhr patent search failed: %s", exc)

        if Fetcher is None:
            return links[:limit]

        # Fallback: rendered HTML via Scrapling.
        search_url = f"https://patents.google.com/?q={quote_plus(query)}&country=US"
        try:
            page = self._fetch_page(search_url)
            html = str(getattr(page, "text", None) or getattr(page, "body", None) or "")
            for url in patent_search.extract_patent_urls_from_html(html, limit=limit):
                if url not in seen:
                    seen.add(url)
                    links.append(url)
            if links:
                return links[:limit]
            if hasattr(page, "css"):
                for selector in ("a[href*='/patent/US']", "a[href*='patent/US']"):
                    try:
                        for item in page.css(selector):
                            href = getattr(item, "attrib", {}).get("href")
                            if not href:
                                continue
                            if href.startswith("/"):
                                href = f"https://patents.google.com{href}"
                            if href not in seen:
                                seen.add(href)
                                links.append(href)
                            if len(links) >= limit:
                                return links
                    except Exception:
                        continue
        except Exception as exc:
            _log.warning("HTML patent search failed: %s", exc)

        return links[:limit]

    def collect_url(self, url: str, source_type: str = "patent_html") -> RawDocument:
        document = self.fetch_raw_document(url, source_type=source_type)
        self.database.add_raw_document(document)
        cur = self.database.connection.execute("SELECT id FROM raw_documents ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        if row:
            try:
                self.database.enqueue_raw_document(int(row["id"]))
            except Exception:
                pass
        return document

    def collect_many(self, urls: list[str], source_type: str = "patent_html") -> list[RawDocument]:
        return [self.collect_url(url, source_type=source_type) for url in urls]

    def parse_document(self, document: RawDocument) -> PatentRecord:
        record = self.parser.parse(document)
        self.database.upsert_patent(record)
        return record

    def parse_all(self) -> list[PatentRecord]:
        parsed_records: list[PatentRecord] = []
        for row in self.database.list_raw_documents():
            document = RawDocument(
                source_url=row["source_url"],
                source_type=row["source_type"],
                fetched_at=datetime.fromisoformat(row["fetched_at"]),
                title=row["title"],
                content_type=row["content_type"],
                raw_text=row["raw_text"],
                raw_html=row["raw_html"],
                metadata=json.loads(row["metadata_json"] or "{}"),
            )
            parsed_records.append(self.parse_document(document))
        return parsed_records

    def process_queue_once(self) -> int:
        """Process a single queue item if available. Returns 1 if processed, 0 if none."""
        item = self.database.get_next_queue_item()
        if item is None:
            return 0
        queue_id = int(item["id"])
        raw_id = int(item["raw_document_id"])
        row = self.database.connection.execute("SELECT * FROM raw_documents WHERE id = ?", (raw_id,)).fetchone()
        if not row:
            self.database.update_queue_status(queue_id, "failed")
            return 0
        document = RawDocument(
            source_url=row["source_url"],
            source_type=row["source_type"],
            fetched_at=datetime.fromisoformat(row["fetched_at"]),
            title=row["title"],
            content_type=row["content_type"],
            raw_text=row["raw_text"],
            raw_html=row["raw_html"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )
        try:
            self.parse_document(document)
            self.database.update_queue_status(queue_id, "done")
            return 1
        except Exception:
            attempts = int(item["attempts"]) + 1
            if attempts >= 3:
                self.database.update_queue_status(queue_id, "failed", attempts=attempts)
            else:
                self.database.update_queue_status(queue_id, "pending", attempts=attempts)
            return 0


def build_reaction_record(
    patent_id: str,
    reaction_number: int,
    reaction_smarts: str | None = None,
    reactant_smiles: list[str] | None = None,
    product_smiles: str | None = None,
    yield_percent: float | None = None,
    temperature_celsius: float | None = None,
    solvent: str | None = None,
    catalyst: str | None = None,
    time_hours: float | None = None,
    mechanism_text: str | None = None,
    notes: str | None = None,
    metadata: dict | None = None,
) -> ReactionRecord:
    return ReactionRecord(
        reaction_id=f"{patent_id}:{reaction_number}",
        patent_id=patent_id,
        reaction_smarts=reaction_smarts,
        reactant_smiles=reactant_smiles or [],
        product_smiles=product_smiles,
        yield_percent=yield_percent,
        temperature_celsius=temperature_celsius,
        solvent=solvent,
        catalyst=catalyst,
        time_hours=time_hours,
        mechanism_text=mechanism_text,
        notes=notes,
        metadata=metadata or {},
    )
