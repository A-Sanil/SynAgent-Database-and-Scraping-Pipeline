"""In-memory crawler status for the review UI (thread-safe enough for single agent)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CrawlerStatus:
    active: bool = False
    profile: str = "dynamic"
    current_query: str | None = None
    last_cycle_at: float | None = None
    urls_found: int = 0
    collected: int = 0
    skipped: int = 0
    errors: int = 0
    fetcher_used: str | None = None
    spider_requests: int = 0
    spider_items: int = 0
    message: str = "Idle"
    recent_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "profile": self.profile,
            "current_query": self.current_query,
            "last_cycle_at": self.last_cycle_at,
            "urls_found": self.urls_found,
            "collected": self.collected,
            "skipped": self.skipped,
            "errors": self.errors,
            "fetcher_used": self.fetcher_used,
            "spider_requests": self.spider_requests,
            "spider_items": self.spider_items,
            "message": self.message,
            "recent_errors": self.recent_errors[-10:],
        }


class CrawlTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status = CrawlerStatus()

    def begin_cycle(self, profile: str, query: str) -> None:
        with self._lock:
            self.status.active = True
            self.status.profile = profile
            self.status.current_query = query
            self.status.message = f"Collecting: {query}"

    def end_cycle(self, stats: dict[str, Any], fetcher: str | None = None) -> None:
        with self._lock:
            self.status.active = False
            self.status.last_cycle_at = time.time()
            self.status.urls_found = int(stats.get("searched", stats.get("urls_found", 0)))
            self.status.collected = int(stats.get("collected", 0))
            self.status.skipped = int(stats.get("skipped", 0))
            self.status.errors = int(stats.get("errors", 0))
            self.status.fetcher_used = fetcher or stats.get("fetcher_used")
            self.status.spider_requests = int(stats.get("spider_requests", 0))
            self.status.spider_items = int(stats.get("spider_items", 0))
            self.status.message = stats.get("message", "Cycle complete")

    def add_error(self, msg: str) -> None:
        with self._lock:
            self.status.errors += 1
            self.status.recent_errors.append(msg)
            if len(self.status.recent_errors) > 20:
                self.status.recent_errors = self.status.recent_errors[-20:]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self.status.to_dict()


tracker = CrawlTracker()
