"""USB-backed file storage helpers (raw HTML, parsed JSON).

The SQLite database uses the same data root via `patent_pipeline.config`.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

from .config import get_parsed_dir, get_raw_dir, init_storage, resolve_data_dir


def init_usb_structure(path: str | None = None) -> str:
    """Create directory layout on the USB drive (or local path)."""
    base = init_storage(path)
    return str(base)


def resolve_data_dir(path: str | None = None) -> str:
    from . import config as _config

    return str(_config.resolve_data_dir(path))


def get_parsed_dir_path(path: str | None = None) -> str:
    return str(get_parsed_dir(path))


def get_raw_dir_path(path: str | None = None) -> str:
    return str(get_raw_dir(path))


def save_parsed_json(patent_id: str, data: dict[str, Any], path: str | None = None) -> str:
    parsed_dir = get_parsed_dir(path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_id = patent_id.replace("/", "_").replace("\\", "_")
    filename = f"patent_{safe_id}_{timestamp}.json"
    filepath = parsed_dir / filename
    temp_path = filepath.with_suffix(filepath.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, filepath)
    return str(filepath)


def save_raw_document(source_url: str, raw_text: str | None, raw_html: str | None, path: str | None = None) -> str | None:
    """Archive fetched raw content beside the database."""
    raw_dir = get_raw_dir(path)
    digest = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16]
    filepath = raw_dir / f"raw_{digest}.json"
    payload = {
        "source_url": source_url,
        "raw_text": raw_text,
        "raw_html": raw_html,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    temp_path = filepath.with_suffix(".json.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, filepath)
    return str(filepath)
