"""Scrapling Spider for US Google Patents chemistry collection."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote_plus

from .crawl_tracker import tracker
from .crawler_registry import CrawlerProfile, get_profile
from .database import PatentDatabase
from .fetchers_bridge import extract_patent_links, google_patents_search_url
from .models import RawDocument
from .pipeline import IngestionPipeline, RawDocumentParserStub

if TYPE_CHECKING:
    from scrapling.engines.toolbelt.custom import Response
    from scrapling.spiders import Request

_log = logging.getLogger("patent_agent.spider")


def _session_kwargs(profile: CrawlerProfile, for_search: bool) -> dict[str, Any]:
    if for_search and profile.name == "stealth":
        return {"headless": True, "solve_cloudflare": True, "network_idle": True}
    return {"headless": True, "network_idle": True}


class PatentChemistrySpider:
    """Scrapling Spider: search pages → patent detail pages → database items."""

    name = "patent_chemistry"
    concurrent_requests = 4
    download_delay = 1.0

    def __init__(
        self,
        queries: list[str],
        limit_per_query: int = 8,
        profile_name: str = "dynamic",
        data_dir: str | Path | None = None,
        crawldir: str | Path | None = None,
    ):
        self.queries = queries
        self.limit_per_query = limit_per_query
        self.profile = get_profile(profile_name)
        self.data_dir = Path(data_dir) if data_dir else None
        self.crawldir = Path(crawldir) if crawldir else None
        self._database: PatentDatabase | None = None
        self._spider_impl: Any = None

    def _build_spider_class(self) -> type:
        profile = self.profile
        queries = self.queries
        limit = self.limit_per_query
        outer = self

        from scrapling.fetchers import AsyncDynamicSession, AsyncStealthySession, FetcherSession
        from scrapling.spiders import Request, Spider

        class _PatentSpider(Spider):
            name = PatentChemistrySpider.name
            concurrent_requests = profile.concurrent_requests
            download_delay = 1.0
            start_urls: list[str] = []

            def configure_sessions(self, manager) -> None:
                if profile.name == "stealth":
                    manager.add(
                        "search",
                        AsyncStealthySession(**_session_kwargs(profile, True)),
                        default=True,
                    )
                    manager.add(
                        "detail",
                        AsyncDynamicSession(**_session_kwargs(profile, False)),
                        lazy=True,
                    )
                else:
                    manager.add(
                        "search",
                        AsyncDynamicSession(**_session_kwargs(profile, True)),
                        default=True,
                    )
                    manager.add("detail", FetcherSession(), lazy=True)

            async def start_requests(self):
                for query in queries:
                    url = google_patents_search_url(query)
                    yield Request(
                        url,
                        sid="search",
                        callback=self.parse_search,
                        meta={"page_type": "search", "query": query},
                    )

            async def parse_search(self, response: "Response"):
                html = str(getattr(response, "text", "") or getattr(response, "html", "") or "")
                links = extract_patent_links(html, limit=limit)
                query = response.meta.get("query", "")
                _log.info("Spider search %r → %s patent links", query, len(links))
                for link in links:
                    if outer._database and outer._database.has_source_url(link):
                        continue
                    yield Request(
                        link,
                        sid="detail",
                        callback=self.parse_patent,
                        meta={"page_type": "patent", "query": query},
                    )

            async def parse_patent(self, response: "Response"):
                title = None
                try:
                    title = response.css("title::text").get()
                except Exception:
                    pass
                yield {
                    "source_url": response.url,
                    "title": title,
                    "raw_text": str(getattr(response, "text", "") or ""),
                    "raw_html": str(getattr(response, "html", "") or ""),
                    "query": response.meta.get("query"),
                    "fetcher": "spider",
                }

            async def parse(self, response: "Response"):
                if response.meta.get("page_type") == "search":
                    async for item in self.parse_search(response):
                        yield item
                else:
                    async for item in self.parse_patent(response):
                        yield item

        return _PatentSpider

    def _get_spider(self) -> Any:
        if self._spider_impl is None:
            cls = self._build_spider_class()
            crawldir = self.crawldir
            if crawldir is None and self.data_dir:
                crawldir = self.data_dir / "spider_checkpoints"
            self._spider_impl = cls(
                crawldir=str(crawldir) if crawldir else None,
                interval=300.0,
            )
            self._spider_impl.concurrent_requests = self.profile.concurrent_requests
        return self._spider_impl

    def run_collect(self, database: PatentDatabase) -> dict[str, int]:
        """Run one spider crawl and persist items to the database."""
        self._database = database
        stats = {"searched": 0, "collected": 0, "skipped": 0, "errors": 0, "spider_items": 0}
        query = self.queries[0] if self.queries else ""
        tracker.begin_cycle(self.profile.name, query)

        try:
            spider = self._get_spider()
            result = spider.start()
            items = result.items or []
            stats["spider_items"] = len(items)
            stats["searched"] = len(items)

            pipeline = IngestionPipeline(database=database, parser=RawDocumentParserStub())
            for item in items:
                url = item.get("source_url")
                if not url:
                    continue
                if database.has_source_url(url):
                    stats["skipped"] += 1
                    continue
                try:
                    doc = RawDocument(
                        source_url=url,
                        source_type="patent_html",
                        fetched_at=datetime.now(timezone.utc),
                        title=item.get("title"),
                        content_type="text/html",
                        raw_text=item.get("raw_text"),
                        raw_html=item.get("raw_html"),
                        metadata={
                            "fetcher": "spider",
                            "query": item.get("query"),
                            "profile": self.profile.name,
                        },
                    )
                    database.add_raw_document(doc)
                    row = database.connection.execute(
                        "SELECT id FROM raw_documents ORDER BY id DESC LIMIT 1"
                    ).fetchone()
                    if row:
                        database.enqueue_raw_document(int(row["id"]))
                    stats["collected"] += 1
                except Exception as exc:
                    stats["errors"] += 1
                    tracker.add_error(str(exc))
                    _log.warning("Save failed %s: %s", url, exc)

            if result.stats:
                stats["spider_requests"] = getattr(result.stats, "requests", 0) or 0
            stats["message"] = f"Spider collected {stats['collected']} patents"
            database.record_crawl_run(
                profile=self.profile.name,
                query=query,
                urls_found=stats.get("spider_requests", 0),
                collected=stats["collected"],
                skipped=stats["skipped"],
                errors=stats["errors"],
                status="paused" if result.paused else "done",
            )
        except Exception as exc:
            stats["errors"] += 1
            stats["message"] = str(exc)
            tracker.add_error(str(exc))
            _log.exception("Spider crawl failed")
            raise
        finally:
            tracker.end_cycle(stats, fetcher="spider")
            self._database = None

        return stats


def run_spider_collect_cycle(
    database: PatentDatabase,
    queries: list[str],
    limit: int = 8,
    profile: str = "dynamic",
) -> dict[str, int]:
    """Convenience entry: run spider for the first query in the list."""
    if not queries:
        return {"searched": 0, "collected": 0, "skipped": 0, "errors": 0}
    spider = PatentChemistrySpider(
        queries=[queries[0]],
        limit_per_query=limit,
        profile_name=profile,
        data_dir=database.data_dir,
    )
    return spider.run_collect(database)
