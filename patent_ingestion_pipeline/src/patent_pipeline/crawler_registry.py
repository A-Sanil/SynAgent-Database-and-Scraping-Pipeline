"""Crawler profile registry: static, dynamic, stealth, spider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CrawlerMode = Literal["static", "dynamic", "stealth", "spider", "auto"]


@dataclass(frozen=True, slots=True)
class CrawlerProfile:
    name: str
    search_mode: str
    detail_mode: str
    use_spider: bool = False
    concurrent_requests: int = 4
    description: str = ""


PROFILES: dict[str, CrawlerProfile] = {
    "static": CrawlerProfile(
        name="static",
        search_mode="static",
        detail_mode="static",
        description="Fast HTTP only (Google Patents search often needs more).",
    ),
    "dynamic": CrawlerProfile(
        name="dynamic",
        search_mode="dynamic",
        detail_mode="dynamic",
        concurrent_requests=4,
        description="Chromium browser automation (recommended for Google Patents).",
    ),
    "stealth": CrawlerProfile(
        name="stealth",
        search_mode="stealth",
        detail_mode="dynamic",
        concurrent_requests=3,
        description="Stealth browser for search, dynamic for patent pages.",
    ),
    "spider": CrawlerProfile(
        name="spider",
        search_mode="dynamic",
        detail_mode="dynamic",
        use_spider=True,
        concurrent_requests=4,
        description="Scrapling Spider with concurrent search + detail crawlers.",
    ),
    "auto": CrawlerProfile(
        name="auto",
        search_mode="auto",
        detail_mode="auto",
        description="Try static, then dynamic, then stealth per request.",
    ),
}


def get_profile(name: str) -> CrawlerProfile:
    key = (name or "dynamic").lower()
    if key not in PROFILES:
        raise ValueError(f"Unknown crawler profile '{name}'. Choose: {', '.join(PROFILES)}")
    return PROFILES[key]
