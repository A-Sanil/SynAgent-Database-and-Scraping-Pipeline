"""PDF and table collector utilities for the patent ingestion pipeline.

This module attempts to extract text, tables and images from PDFs. It uses
`pdfplumber` for text extraction and `camelot` for table extraction when
available.  Functions are defensive and return as much as possible even if
optional dependencies are missing.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import asdict
from typing import Any

import requests

try:
    import pdfplumber
except Exception:  # pragma: no cover - optional
    pdfplumber = None

try:
    import camelot
except Exception:  # pragma: no cover - optional
    camelot = None


def download_url_to_bytes(url: str) -> bytes:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def extract_text_from_pdf_bytes(data: bytes) -> str:
    if pdfplumber is None:
        # fallback: return empty string
        return ""
    text_parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                text_parts.append("")
    return "\n\n".join(text_parts)


def extract_tables_from_pdf_bytes(data: bytes) -> list[dict[str, Any]]:
    # camelot needs a filename; write to temp file
    if camelot is None:
        return []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(data)
        tmpname = tf.name
    tables = []
    try:
        tables_found = camelot.read_pdf(tmpname, pages="all")
        for t in tables_found:
            tables.append({"df": t.df.to_json(orient="split"), "shape": t.shape})
    except Exception:
        tables = []
    finally:
        try:
            os.unlink(tmpname)
        except Exception:
            pass
    return tables


def collect_pdf_from_url(url: str) -> dict[str, Any]:
    """Download a PDF URL and produce a dict similar to RawDocument metadata.

    Returns a dict with keys: raw_text, raw_tables (list), raw_bytes (bytes length), metadata
    """
    data = download_url_to_bytes(url)
    text = extract_text_from_pdf_bytes(data)
    tables = extract_tables_from_pdf_bytes(data)
    return {
        "raw_text": text,
        "raw_tables": tables,
        "raw_bytes_len": len(data),
        "metadata": {"source_url": url, "collector": "pdfplumber/camelot"},
    }
