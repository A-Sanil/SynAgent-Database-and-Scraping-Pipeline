"""Ingest a bulk patent CSV file into the pipeline database.

This module handles the "drop a CSV into the UI" workflow:
  1. Read each row from the CSV (produced by get_bulk_patent_data or compatible).
  2. Create a RawDocument per row and store it in raw_documents.
  3. Enqueue each document for LLM parsing by the background worker.

Compatible CSV columns (extras ignored):
    patent_id, title, abstract, date, url, raw_text
    (also accepts: patent_title, patent_abstract, grant_date, source_url, text, full_text)
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .database import PatentDatabase
from .models import RawDocument


def ingest_csv_file(
    csv_path: Path | str,
    db: PatentDatabase,
    enqueue: bool = True,
    skip_existing: bool = True,
    verbose: bool = True,
) -> int:
    """Read a patent CSV file and store each row as a RawDocument ready for LLM parsing.

    Args:
        csv_path: Path to the CSV file (UTF-8 or UTF-8-BOM).
        db: Open PatentDatabase instance.
        enqueue: Automatically add each document to the parse queue.
        skip_existing: Skip rows whose source_url already exists in the database.
        verbose: Print progress.

    Returns:
        Number of documents ingested.
    """
    from .bulk_downloader import load_csv_as_patent_dicts

    path = Path(csv_path)
    records = load_csv_as_patent_dicts(path)

    if verbose:
        print(f"[CSV] Loaded {len(records):,} rows from {path.name}")

    inserted = 0
    skipped = 0

    for rec in records:
        source_url = rec.get("url") or rec.get("patent_id") or str(inserted)
        if skip_existing and db.has_source_url(source_url):
            skipped += 1
            continue

        doc = _record_to_raw_document(rec, source_url)
        db.add_raw_document(doc)

        if enqueue:
            row = db.connection.execute(
                "SELECT id FROM raw_documents WHERE source_url = ? ORDER BY id DESC LIMIT 1",
                (source_url,),
            ).fetchone()
            if row:
                try:
                    db.enqueue_raw_document(int(row["id"]))
                except Exception:
                    pass

        inserted += 1
        if verbose and inserted % 100 == 0:
            print(f"[CSV]   {inserted:,} documents ingested …")

    if verbose:
        print(f"[CSV] Done. {inserted:,} ingested, {skipped:,} skipped (already in DB).")

    return inserted


def ingest_csv_bytes(
    content: bytes,
    filename: str,
    db: PatentDatabase,
    enqueue: bool = True,
    skip_existing: bool = True,
) -> int:
    """Ingest a CSV from raw bytes (used by the FastAPI upload endpoint).

    Args:
        content: Raw file bytes (UTF-8 or UTF-8-BOM).
        filename: Original filename (used only for logging).
        db: Open PatentDatabase instance.
        enqueue: Automatically enqueue each document for parsing.
        skip_existing: Skip rows whose URL already exists.

    Returns:
        Number of documents ingested.
    """
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    records = _rows_to_dicts(reader)

    inserted = 0
    skipped = 0

    for rec in records:
        source_url = rec.get("url") or rec.get("patent_id") or f"csv-upload:{filename}:{inserted}"
        if skip_existing and db.has_source_url(source_url):
            skipped += 1
            continue

        doc = _record_to_raw_document(rec, source_url)
        db.add_raw_document(doc)

        if enqueue:
            row = db.connection.execute(
                "SELECT id FROM raw_documents WHERE source_url = ? ORDER BY id DESC LIMIT 1",
                (source_url,),
            ).fetchone()
            if row:
                try:
                    db.enqueue_raw_document(int(row["id"]))
                except Exception:
                    pass

        inserted += 1

    return inserted


def _rows_to_dicts(reader: csv.DictReader) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in reader:
        raw_text = (
            row.get("raw_text") or row.get("full_text") or row.get("text") or ""
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


def _record_to_raw_document(rec: dict[str, Any], source_url: str) -> RawDocument:
    patent_id = rec.get("patent_id") or ""
    title = rec.get("title") or ""
    raw_text = rec.get("raw_text") or f"{title}\n\n{rec.get('abstract', '')}".strip()
    return RawDocument(
        source_url=source_url,
        source_type="patent_csv",
        fetched_at=datetime.now(tz=timezone.utc),
        title=title or None,
        content_type="text/plain",
        raw_text=raw_text or None,
        raw_html=None,
        metadata={
            "patent_id": patent_id,
            "grant_date": rec.get("date") or "",
            "ingest_source": "csv_upload",
        },
    )
