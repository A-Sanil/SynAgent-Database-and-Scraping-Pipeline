"""Central path configuration for local dev and USB-backed storage."""

from __future__ import annotations

import os
import string
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

USB_MARKER = ".synagent_usb"
DEFAULT_LOCAL_DATA = Path("data")
DEFAULT_USB_FOLDER_NAMES = ("synagent_patent_data", "patent_data", "SynAgent")


def resolve_data_dir(explicit: str | Path | None = None) -> Path:
    """Return the patent data root (raw/, parsed/, db/ live under this)."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()

    env_dir = os.environ.get("PATENT_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    detected = detect_usb_data_dir()
    if detected is not None:
        return detected

    return (Path.cwd() / DEFAULT_LOCAL_DATA).resolve()


def detect_usb_data_dir() -> Path | None:
    """Pick a removable drive that already has (or can host) SynAgent patent data."""
    if os.name != "nt":
        return _detect_usb_posix()
    return _detect_usb_windows()


def _detect_usb_windows() -> Path | None:
    try:
        import ctypes

        drive_mask = ctypes.windll.kernel32.GetLogicalDrives()
    except Exception:
        return None

    for index, letter in enumerate(string.ascii_uppercase):
        if not (drive_mask & (1 << index)):
            continue
        root = Path(f"{letter}:\\")
        try:
            drive_type = ctypes.windll.kernel32.GetDriveTypeW(str(root))
        except Exception:
            continue
        # DRIVE_REMOVABLE == 2
        if drive_type != 2:
            continue
        for folder_name in DEFAULT_USB_FOLDER_NAMES:
            candidate = root / folder_name
            if (candidate / USB_MARKER).exists() or (candidate / "db").exists():
                return candidate.resolve()
        # Prefer an empty synagent folder on the USB if nothing exists yet.
        preferred = root / DEFAULT_USB_FOLDER_NAMES[0]
        if preferred.parent.exists():
            return preferred.resolve()
    return None


def _detect_usb_posix() -> Path | None:
    media_root = Path("/media")
    if not media_root.exists():
        return None
    for mount in sorted(media_root.iterdir(), reverse=True):
        if not mount.is_dir():
            continue
        for folder_name in DEFAULT_USB_FOLDER_NAMES:
            candidate = mount / folder_name
            if (candidate / USB_MARKER).exists() or (candidate / "db").exists():
                return candidate.resolve()
    return None


def init_storage(data_dir: str | Path | None = None) -> Path:
    """Create raw/, parsed/, db/ and mark the directory as SynAgent USB storage."""
    base = resolve_data_dir(data_dir)
    for sub in ("raw", "parsed", "db"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    marker = base / USB_MARKER
    marker.write_text("SynAgent patent ingestion data\n", encoding="utf-8")
    return base


def get_db_path(data_dir: str | Path | None = None) -> Path:
    base = resolve_data_dir(data_dir)
    return base / "db" / "patent_pipeline.db"


def get_raw_dir(data_dir: str | Path | None = None) -> Path:
    return resolve_data_dir(data_dir) / "raw"


def get_parsed_dir(data_dir: str | Path | None = None) -> Path:
    return resolve_data_dir(data_dir) / "parsed"
