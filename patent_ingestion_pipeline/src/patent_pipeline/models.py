"""Core data models for the separate patent ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class RawDocument:
    source_url: str
    source_type: str
    fetched_at: datetime
    title: str | None = None
    content_type: str | None = None
    raw_text: str | None = None
    raw_html: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReactionRecord:
    reaction_id: str
    patent_id: str
    reaction_smarts: str | None = None
    reactant_smiles: list[str] = field(default_factory=list)
    product_smiles: str | None = None
    yield_percent: float | None = None
    temperature_celsius: float | None = None
    solvent: str | None = None
    catalyst: str | None = None
    time_hours: float | None = None
    mechanism_text: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PatentRecord:
    patent_id: str
    title: str
    abstract: str | None = None
    source_url: str | None = None
    publication_date: str | None = None
    inventors: list[str] = field(default_factory=list)
    assignee: str | None = None
    domain_tags: list[str] = field(default_factory=list)
    target_terms: list[str] = field(default_factory=list)
    raw_text: str | None = None
    reactions: list[ReactionRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
