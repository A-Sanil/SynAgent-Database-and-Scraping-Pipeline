"""Download bulk chemistry patent data from the PatentsView public API.

Usage:
    from patent_pipeline.bulk_downloader import get_bulk_patent_data

    records = get_bulk_patent_data(year=2024, week=1, limit=500, output_csv="patents_2024_w1.csv")
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

PATENTSVIEW_API = "https://search.patentsview.org/api/v1/patent/"

# CPC section C = Chemistry and Metallurgy (C01–C12 are the main organic/inorganic groups)
_CHEMISTRY_CPC_PREFIXES = [
    "C01", "C02", "C03", "C04", "C05", "C06",
    "C07", "C08", "C09", "C10", "C11", "C12",
]

CSV_FIELDS = ["patent_id", "title", "abstract", "date", "url", "raw_text"]


def _week_monday(year: int, week: int) -> datetime:
    """Return the Monday datetime for an ISO week (week 1 = first week with a Thursday)."""
    return datetime.strptime(f"{year}-W{week:02d}-1", "%Y-W%W-%w")


def _week_date_range(year: int, week: int) -> tuple[str, str]:
    """Return (start_date, end_date) as YYYY-MM-DD strings for the given ISO week."""
    monday = _week_monday(year, week)
    sunday = monday + timedelta(days=6)
    return monday.strftime("%Y-%m-%d"), sunday.strftime("%Y-%m-%d")


def get_bulk_patent_data(
    year: int = 2024,
    week: int = 1,
    limit: int = 500,
    output_csv: Path | str | None = None,
    chemistry_only: bool = True,
) -> list[dict[str, Any]]:
    """Fetch weekly patent metadata from PatentsView and optionally save as CSV.

    Args:
        year: USPTO grant year (PatentsView covers 1976+; full-text ~2005+).
        week: ISO week number (1–53).
        limit: Maximum patents to return (PatentsView caps at 10 000 per request).
        output_csv: If provided, write results to this path as a UTF-8 CSV.
        chemistry_only: When True, filter to CPC section C patents only.

    Returns:
        List of dicts with keys: patent_id, title, abstract, date, url, raw_text.

    Raises:
        RuntimeError: If the 'requests' package is missing or the API call fails.
    """
    try:
        import requests
    except ImportError:
        raise RuntimeError("The 'requests' package is required: pip install requests")

    start_date, end_date = _week_date_range(year, week)

    and_clauses: list[dict] = [
        {"_gte": {"grant_date": start_date}},
        {"_lte": {"grant_date": end_date}},
    ]
    if chemistry_only:
        and_clauses.append(
            {"_or": [{"cpc_group_id": f"{pfx}"} for pfx in _CHEMISTRY_CPC_PREFIXES]}
        )

    body: dict[str, Any] = {
        "q": {"_and": and_clauses},
        "f": ["patent_id", "patent_title", "patent_abstract", "grant_date"],
        "s": [{"grant_date": "asc"}],
        "o": {"per_page": min(limit, 1000), "matched_subentities_only": False},
    }

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    try:
        resp = requests.post(PATENTSVIEW_API, json=body, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"PatentsView API request failed: {exc}") from exc

    raw_patents: list[dict] = data.get("patents") or []
    results: list[dict[str, Any]] = []

    for p in raw_patents[:limit]:
        pid = str(p.get("patent_id") or "")
        title = str(p.get("patent_title") or "")
        abstract = str(p.get("patent_abstract") or "")
        date = str(p.get("grant_date") or "")
        url = f"https://patents.google.com/patent/US{pid}/en" if pid else ""
        raw_text = f"{title}\n\n{abstract}".strip()
        results.append({
            "patent_id": pid,
            "title": title,
            "abstract": abstract,
            "date": date,
            "url": url,
            "raw_text": raw_text,
        })

    if output_csv is not None:
        _write_csv(results, Path(output_csv))

    return results


def load_csv_as_patent_dicts(csv_path: Path | str) -> list[dict[str, Any]]:
    """Read a CSV file (produced by get_bulk_patent_data or compatible format) into patent dicts.

    Expected columns (extras are ignored): patent_id, title, abstract, date, url, raw_text.
    Any column named 'text' or 'full_text' is also accepted as raw_text.
    """
    path = Path(csv_path)
    records: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_text = (
                row.get("raw_text")
                or row.get("full_text")
                or row.get("text")
                or ""
            )
            if not raw_text:
                title = row.get("title") or row.get("patent_title") or ""
                abstract = row.get("abstract") or row.get("patent_abstract") or ""
                raw_text = f"{title}\n\n{abstract}".strip()
            records.append({
                "patent_id": row.get("patent_id") or row.get("id") or "",
                "title": row.get("title") or row.get("patent_title") or "",
                "abstract": row.get("abstract") or row.get("patent_abstract") or "",
                "date": row.get("date") or row.get("grant_date") or row.get("publication_date") or "",
                "url": row.get("url") or row.get("source_url") or "",
                "raw_text": raw_text,
            })
    return records


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
