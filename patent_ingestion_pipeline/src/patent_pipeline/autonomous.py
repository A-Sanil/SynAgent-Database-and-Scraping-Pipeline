"""Autonomous patent ingestion: collect US chemistry patents, parse, build USB database."""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .config import init_storage, resolve_data_dir
from .crawl_tracker import tracker
from .crawler_registry import get_profile
from .database import PatentDatabase
from .llm_parser import GeminiLLMParser
from .patent_spider import run_spider_collect_cycle
from .pipeline import IngestionPipeline, RawDocumentParserStub

load_dotenv()


def _package_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]


DEFAULT_QUERIES = [
    "organic synthesis catalyst",
    "pharmaceutical SMILES reaction",
    "chemical process yield temperature",
    "battery electrolyte synthesis",
    "heterocyclic compound preparation",
]

_log = logging.getLogger("patent_agent")


@dataclass
class AgentConfig:
    data_dir: Path
    queries: list[str] = field(default_factory=lambda: list(DEFAULT_QUERIES))
    patents_per_query: int = 8
    parse_batch_per_cycle: int = 5
    collect_sleep_seconds: float = 120.0
    parse_idle_seconds: float = 3.0
    model: str = "gemini-2.0-flash"
    api_key: str | None = None
    with_ui: bool = True
    ui_host: str = "127.0.0.1"
    ui_port: int = 8001
    queries_file: Path | None = None
    crawler: str = "dynamic"

    @classmethod
    def from_env(cls, data_dir: Path | None = None) -> AgentConfig:
        base = resolve_data_dir(data_dir)
        queries_path = os.environ.get("PATENT_QUERIES_FILE")
        default_queries = _package_root() / "data" / "chemistry_queries.txt"
        queries_file = Path(queries_path) if queries_path else default_queries
        if not queries_file.is_file():
            queries_file = default_queries
        queries = load_queries(queries_file) if queries_file.is_file() else list(DEFAULT_QUERIES)
        return cls(
            data_dir=base,
            queries=queries,
            patents_per_query=int(os.environ.get("PATENT_COLLECT_LIMIT", "8")),
            parse_batch_per_cycle=int(os.environ.get("PATENT_PARSE_BATCH", "5")),
            collect_sleep_seconds=float(os.environ.get("PATENT_COLLECT_SLEEP", "120")),
            parse_idle_seconds=float(os.environ.get("PATENT_PARSE_IDLE", "3")),
            model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
            api_key=os.environ.get("GEMINI_API_KEY"),
            with_ui=os.environ.get("PATENT_AGENT_UI", "1") not in ("0", "false", "False"),
            ui_host=os.environ.get("PATENT_UI_HOST", "127.0.0.1"),
            ui_port=int(os.environ.get("PATENT_UI_PORT", "8001")),
            queries_file=queries_file if queries_file.is_file() else None,
            crawler=os.environ.get("PATENT_CRAWLER", "dynamic"),
        )


def load_queries(path: Path) -> list[str]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return lines or list(DEFAULT_QUERIES)


def setup_logging(data_dir: Path) -> None:
    log_path = data_dir / "agent.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _state_path(data_dir: Path) -> Path:
    return data_dir / "agent_state.json"


def load_query_index(data_dir: Path, n_queries: int) -> int:
    path = _state_path(data_dir)
    if not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("query_index", 0)) % max(n_queries, 1)
    except Exception:
        return 0


def save_query_index(data_dir: Path, index: int) -> None:
    path = _state_path(data_dir)
    path.write_text(json.dumps({"query_index": index, "updated_at": time.time()}), encoding="utf-8")


class AutonomousAgent:
    """Runs forever: search → download → enqueue → parse → repeat."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._stop = threading.Event()
        init_storage(config.data_dir)
        setup_logging(config.data_dir)

    def stop(self) -> None:
        self._stop.set()

    def _open_collect_pipeline(self) -> IngestionPipeline:
        db = PatentDatabase(data_dir=self.config.data_dir)
        return IngestionPipeline(database=db, parser=RawDocumentParserStub())

    def _open_parse_pipeline(self) -> IngestionPipeline:
        db = PatentDatabase(data_dir=self.config.data_dir)
        parser = GeminiLLMParser(api_key=self.config.api_key, model=self.config.model)
        return IngestionPipeline(database=db, parser=parser)

    def collect_cycle(self) -> dict[str, int]:
        cfg = self.config
        stats = {"searched": 0, "collected": 0, "skipped": 0, "errors": 0}
        if not cfg.queries:
            _log.warning("No search queries configured")
            return stats

        idx = load_query_index(cfg.data_dir, len(cfg.queries))
        query = cfg.queries[idx]
        save_query_index(cfg.data_dir, (idx + 1) % len(cfg.queries))

        _log.info("Collect cycle — query: %s (crawler=%s)", query, cfg.crawler)
        profile = get_profile(cfg.crawler)
        tracker.begin_cycle(profile.name, query)

        if profile.use_spider:
            db = PatentDatabase(data_dir=cfg.data_dir)
            try:
                stats = run_spider_collect_cycle(
                    db,
                    queries=[query],
                    limit=cfg.patents_per_query,
                    profile=profile.name,
                )
                db.enqueue_unqueued_raw_documents()
            finally:
                db.close()
            return stats

        pipeline = self._open_collect_pipeline()
        fetcher_used: str | None = None
        try:
            from .fetchers_bridge import search_patent_urls

            urls, fetcher_used = search_patent_urls(
                query, limit=cfg.patents_per_query, mode=profile.search_mode
            )
            stats["searched"] = len(urls)
            for url in urls:
                if self._stop.is_set():
                    break
                try:
                    if pipeline.database.has_source_url(url):
                        stats["skipped"] += 1
                        continue
                    pipeline.collect_url(url)
                    stats["collected"] += 1
                    _log.info("Collected %s", url)
                except Exception as exc:
                    stats["errors"] += 1
                    tracker.add_error(str(exc))
                    _log.warning("Collect failed %s: %s", url, exc)
            enqueued = pipeline.database.enqueue_unqueued_raw_documents()
            pipeline.database.record_crawl_run(
                profile=profile.name,
                query=query,
                urls_found=stats["searched"],
                collected=stats["collected"],
                skipped=stats["skipped"],
                errors=stats["errors"],
            )
            stats["fetcher_used"] = fetcher_used
            _log.info(
                "Collect done — new=%s skipped=%s enqueued=%s fetcher=%s",
                stats["collected"],
                stats["skipped"],
                enqueued,
                fetcher_used,
            )
        finally:
            pipeline.database.close()
        tracker.end_cycle(stats, fetcher=fetcher_used)
        return stats

    def parse_cycle(self) -> int:
        cfg = self.config
        processed = 0
        pipeline = self._open_parse_pipeline()
        try:
            for _ in range(cfg.parse_batch_per_cycle):
                if self._stop.is_set():
                    break
                try:
                    n = pipeline.process_queue_once()
                except Exception as exc:
                    _log.warning("Parse error: %s", exc)
                    break
                if n == 0:
                    break
                processed += n
                _log.info("Parsed one patent from queue")
        finally:
            pipeline.database.close()
        return processed

    def log_stats(self) -> None:
        db = PatentDatabase(data_dir=self.config.data_dir)
        try:
            patents = db.connection.execute("SELECT COUNT(*) FROM patents").fetchone()[0]
            reactions = db.connection.execute("SELECT COUNT(*) FROM reactions").fetchone()[0]
            raw = db.connection.execute("SELECT COUNT(*) FROM raw_documents").fetchone()[0]
            pending = db.connection.execute(
                "SELECT COUNT(*) FROM parse_queue WHERE status = 'pending'"
            ).fetchone()[0]
            _log.info(
                "Database — patents=%s reactions=%s raw=%s queue_pending=%s path=%s",
                patents,
                reactions,
                raw,
                pending,
                db.db_path,
            )
        finally:
            db.close()

    def start_ui(self) -> None:
        if not self.config.with_ui:
            return
        os.environ["PATENT_DATA_DIR"] = str(self.config.data_dir)

        def _run() -> None:
            import uvicorn

            from .webui import app as webui_app

            uvicorn.run(webui_app, host=self.config.ui_host, port=self.config.ui_port, log_level="warning")

        thread = threading.Thread(target=_run, daemon=True, name="review-ui")
        thread.start()
        _log.info("Review UI: http://%s:%s/", self.config.ui_host, self.config.ui_port)

    def run(self) -> None:
        cfg = self.config
        _log.info("SynAgent patent agent started")
        _log.info("Data directory: %s", cfg.data_dir)
        _log.info("Queries loaded: %s", len(cfg.queries))
        if not cfg.api_key:
            _log.warning("GEMINI_API_KEY not set — parse step will fail until you add it to .env")

        self.start_ui()

        def _handle_signal(*_args: object) -> None:
            _log.info("Shutdown requested…")
            self.stop()

        signal.signal(signal.SIGINT, _handle_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _handle_signal)

        while not self._stop.is_set():
            self.collect_cycle()
            parsed = self.parse_cycle()
            self.log_stats()
            if parsed == 0 and not self._stop.is_set():
                _log.info("Idle %ss — waiting for next collect cycle", cfg.collect_sleep_seconds)
                self._stop.wait(cfg.collect_sleep_seconds)
            elif not self._stop.is_set():
                self._stop.wait(cfg.parse_idle_seconds)

        _log.info("Agent stopped")


def run_autonomous(
    data_dir: Optional[Path] = None,
    queries_file: Optional[Path] = None,
    patents_per_query: int = 8,
    parse_batch: int = 5,
    collect_sleep: float = 120.0,
    with_ui: bool = True,
    ui_port: int = 8001,
    model: str = "gemini-2.0-flash",
    api_key: str | None = None,
    crawler: str = "dynamic",
) -> None:
    cfg = AgentConfig.from_env(data_dir)
    if queries_file is not None:
        cfg.queries = load_queries(queries_file)
        cfg.queries_file = queries_file
    cfg.patents_per_query = patents_per_query
    cfg.parse_batch_per_cycle = parse_batch
    cfg.collect_sleep_seconds = collect_sleep
    cfg.with_ui = with_ui
    cfg.ui_port = ui_port
    cfg.model = model
    if api_key:
        cfg.api_key = api_key
    cfg.crawler = crawler
    AutonomousAgent(cfg).run()


def main() -> None:
    """Entry point for SynAgentPatent.exe (no typer required)."""
    import argparse

    parser = argparse.ArgumentParser(description="SynAgent autonomous US patent chemistry agent")
    parser.add_argument("--data-dir", type=str, default=None, help="USB folder, e.g. D:\\synagent_patent_data")
    parser.add_argument("--queries-file", type=str, default=None, help="Text file with one search query per line")
    parser.add_argument("--limit", type=int, default=8, help="Patents per search query per cycle")
    parser.add_argument("--parse-batch", type=int, default=5, help="Max patents to parse per cycle")
    parser.add_argument("--sleep", type=float, default=120.0, help="Seconds between collect cycles when idle")
    parser.add_argument("--no-ui", action="store_true", help="Do not start the review web UI")
    parser.add_argument("--port", type=int, default=8001, help="Review UI port")
    parser.add_argument(
        "--crawler",
        type=str,
        default="dynamic",
        choices=["static", "dynamic", "stealth", "spider", "auto"],
        help="Scrapling profile: dynamic (browser), stealth, spider (concurrent)",
    )
    args = parser.parse_args()

    data = Path(args.data_dir) if args.data_dir else None
    qf = Path(args.queries_file) if args.queries_file else None
    if data is not None:
        os.environ["PATENT_DATA_DIR"] = str(data.resolve())

    run_autonomous(
        data_dir=data,
        queries_file=qf,
        patents_per_query=args.limit,
        parse_batch=args.parse_batch,
        collect_sleep=args.sleep,
        with_ui=not args.no_ui,
        ui_port=args.port,
        crawler=args.crawler,
    )


if __name__ == "__main__":
    main()
