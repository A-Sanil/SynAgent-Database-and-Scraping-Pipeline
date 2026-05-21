"""Command-line interface for the separate patent ingestion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

from .config import detect_usb_data_dir, get_db_path, init_storage, resolve_data_dir
from .database import PatentDatabase
from .llm_parser import GeminiLLMParser
from .pipeline import IngestionPipeline, RawDocumentParserStub

load_dotenv()

app = typer.Typer(help="Separate patent ingestion pipeline")


def _open_db(db_path: Optional[Path], data_dir: Optional[Path]) -> PatentDatabase:
    if db_path is not None:
        return PatentDatabase(db_path=db_path, data_dir=data_dir)
    return PatentDatabase(data_dir=data_dir)


@app.command("init-db")
def init_db(
    db_path: Optional[Path] = typer.Option(None, help="Explicit SQLite file path"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="USB or local data root"),
) -> None:
    """Initialize SQLite schema under the data directory."""
    base = init_storage(data_dir)
    db = _open_db(db_path, base)
    path = db.db_path
    db.close()
    typer.echo(f"Initialized database at {path}")


@app.command("init-usb")
def init_usb(
    drive: Optional[Path] = typer.Option(
        None,
        help="USB root (e.g. E:\\). Defaults to PATENT_DATA_DIR or auto-detected removable drive.",
    ),
) -> None:
    """Create synagent_patent_data on a USB drive and print the path to set in .env."""
    if drive is not None:
        base = init_storage(drive / "synagent_patent_data")
    else:
        detected = detect_usb_data_dir()
        if detected is not None:
            base = init_storage(detected)
        else:
            removable_root = typer.prompt("No USB auto-detected. Enter drive root (e.g. E:\\)")
            base = init_storage(Path(removable_root) / "synagent_patent_data")
    db = PatentDatabase(data_dir=base)
    db.close()
    typer.echo(f"USB storage ready at: {base}")
    typer.echo(f"Add to .env: PATENT_DATA_DIR={base}")


@app.command()
def status(data_dir: Optional[Path] = typer.Option(None, "--data-dir")) -> None:
    """Show resolved storage paths (useful before launch)."""
    base = resolve_data_dir(data_dir)
    db_path = get_db_path(base)
    typer.echo(f"data_dir: {base}")
    typer.echo(f"database: {db_path}")
    typer.echo(f"raw:      {base / 'raw'}")
    typer.echo(f"parsed:   {base / 'parsed'}")


@app.command()
def collect(
    url: str,
    db_path: Optional[Path] = typer.Option(None, "--db"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
) -> None:
    db = _open_db(db_path, data_dir)
    pipeline = IngestionPipeline(database=db, parser=RawDocumentParserStub())
    document = pipeline.collect_url(url)
    db.close()
    typer.echo(f"Collected raw document from: {document.source_url}")


@app.command("collect-many")
def collect_many(
    urls_file: Path,
    db_path: Optional[Path] = typer.Option(None, "--db"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
) -> None:
    urls = [line.strip() for line in urls_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    db = _open_db(db_path, data_dir)
    pipeline = IngestionPipeline(database=db, parser=RawDocumentParserStub())
    documents = pipeline.collect_many(urls)
    db.close()
    typer.echo(f"Collected {len(documents)} raw documents")


@app.command("collect-search")
def collect_search(
    query: str,
    limit: int = 10,
    db_path: Optional[Path] = typer.Option(None, "--db"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    crawler: str = typer.Option("dynamic", help="Fetch profile: dynamic | stealth | auto"),
) -> None:
    """Search Google Patents for US results via Scrapling and collect the pages."""
    from .crawler_registry import get_profile

    from .fetchers_bridge import search_patent_urls

    db = _open_db(db_path, data_dir)
    profile = get_profile(crawler)
    pipeline = IngestionPipeline(database=db, parser=RawDocumentParserStub())
    urls, fetcher = search_patent_urls(query, limit=limit, mode=profile.search_mode)
    typer.echo(f"Search fetcher: {fetcher}")
    if not urls:
        db.close()
        typer.echo(f"No patent URLs found for query: {query}")
        raise typer.Exit(code=1)
    documents = pipeline.collect_many(urls)
    db.close()
    typer.echo(f"Collected {len(documents)} patent pages for query: {query}")


@app.command()
def parse(
    db_path: Optional[Path] = typer.Option(None, "--db"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    api_key: str | None = typer.Option(None, help="Google Gemini API key (falls back to GEMINI_API_KEY)"),
    model: str = typer.Option("gemini-2.0-flash", help="Gemini model name"),
    parser: str = typer.Option("gemini", help="Parser backend: gemini"),
) -> None:
    db = _open_db(db_path, data_dir)
    if parser.lower() != "gemini":
        raise typer.BadParameter("Only the Gemini parser is supported now.")
    pipeline = IngestionPipeline(database=db, parser=GeminiLLMParser(api_key=api_key, model=model))
    records = pipeline.parse_all()
    db.close()
    typer.echo(f"Parsed {len(records)} patent records")


@app.command()
def runserver(
    host: str = "127.0.0.1",
    port: int = 8001,
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
) -> None:
    """Run the human-in-the-loop review UI."""
    import os

    import uvicorn

    if data_dir is not None:
        os.environ["PATENT_DATA_DIR"] = str(resolve_data_dir(data_dir))
    from .webui import app as webui_app

    typer.echo(f"Review UI: http://{host}:{port}/")
    typer.echo(f"Data dir:  {resolve_data_dir()}")
    uvicorn.run(webui_app, host=host, port=port)


@app.command("enqueue-all")
def enqueue_all(
    db_path: Optional[Path] = typer.Option(None, "--db"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
) -> None:
    """Enqueue all raw documents that are not yet queued."""
    db = _open_db(db_path, data_dir)
    cur = db.connection.execute("SELECT id FROM raw_documents ORDER BY id")
    count = 0
    for r in cur.fetchall():
        try:
            db.enqueue_raw_document(int(r["id"]))
            count += 1
        except Exception:
            pass
    db.close()
    typer.echo(f"Enqueued {count} raw documents")


@app.command("run-worker")
def run_worker(
    db_path: Optional[Path] = typer.Option(None, "--db"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    model: str = "gemini-2.0-flash",
    api_key: str | None = typer.Option(None, help="Google Gemini API key"),
    interval: float = 2.0,
) -> None:
    """Process the parse queue continuously."""
    import time

    db = _open_db(db_path, data_dir)
    pipeline = IngestionPipeline(database=db, parser=GeminiLLMParser(api_key=api_key, model=model))
    typer.echo(f"Worker using database: {db.db_path}")
    typer.echo("Starting worker; press Ctrl+C to stop")
    try:
        while True:
            processed = pipeline.process_queue_once()
            if processed == 0:
                time.sleep(interval)
    except KeyboardInterrupt:
        typer.echo("Worker stopped")
    finally:
        db.close()


@app.command("run-crawler")
def run_crawler(
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    queries_file: Optional[Path] = typer.Option(None, "--queries-file"),
    limit: int = 8,
    crawler: str = typer.Option(
        "dynamic",
        help="Profile: static | dynamic | stealth | spider | auto",
    ),
    cycles: int = typer.Option(0, help="Repeat N collect cycles (0 = run once)"),
) -> None:
    """Run one or more collect cycles using Scrapling browser/spider crawlers."""
    import os
    import time

    from .autonomous import AutonomousAgent, AgentConfig, load_queries
    from .config import resolve_data_dir

    if data_dir is not None:
        os.environ["PATENT_DATA_DIR"] = str(resolve_data_dir(data_dir))
    cfg = AgentConfig.from_env(data_dir)
    cfg.crawler = crawler
    if queries_file:
        cfg.queries = load_queries(queries_file)
    cfg.patents_per_query = limit
    agent = AutonomousAgent(cfg)
    n = max(cycles, 1)
    for i in range(n):
        typer.echo(f"Crawler cycle {i + 1}/{n} ({crawler})")
        agent.collect_cycle()
        if cycles == 0 or i + 1 >= n:
            break
        time.sleep(5)


@app.command("test-search")
def test_search(
    query: str,
    limit: int = 5,
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
) -> None:
    """Verify patent search returns US URLs (xhr + fallbacks)."""
    from .pipeline import IngestionPipeline, RawDocumentParserStub

    db = _open_db(None, data_dir)
    pipe = IngestionPipeline(database=db, parser=RawDocumentParserStub())
    urls = pipe.search_google_patents(query=query, limit=limit)
    typer.echo(f"Found {len(urls)} patent URL(s) for: {query}")
    for url in urls:
        typer.echo(f"  {url}")
    if urls:
        doc = pipe.fetch_raw_document(urls[0])
        typer.echo(f"Sample raw text length: {len(doc.raw_text or '')} chars ({doc.source_type})")
    db.close()
    if not urls:
        raise typer.Exit(code=1)


@app.command("run-autonomous")
def run_autonomous_cmd(
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="USB data root"),
    queries_file: Optional[Path] = typer.Option(
        None,
        "--queries-file",
        help="One chemistry search query per line (default: data/chemistry_queries.txt)",
    ),
    limit: int = typer.Option(8, help="US patents to collect per query each cycle"),
    parse_batch: int = typer.Option(5, help="Patents to parse per cycle"),
    sleep: float = typer.Option(120.0, help="Seconds between cycles when queue is empty"),
    with_ui: bool = typer.Option(True, "--ui/--no-ui", help="Start review UI in background"),
    port: int = typer.Option(8001, help="Review UI port"),
    model: str = "gemini-2.0-flash",
    api_key: str | None = typer.Option(None, help="Gemini API key"),
    crawler: str = typer.Option(
        "dynamic",
        help="Scrapling profile: dynamic | stealth | spider | auto",
    ),
) -> None:
    """Run continuously: Scrapling collect → enqueue → Gemini parse → USB database."""
    import os

    from .autonomous import run_autonomous

    if data_dir is not None:
        os.environ["PATENT_DATA_DIR"] = str(resolve_data_dir(data_dir))
    typer.echo("Autonomous agent starting — Ctrl+C to stop")
    typer.echo(f"Data: {resolve_data_dir(data_dir)}")
    typer.echo(f"Review UI: http://127.0.0.1:{port}/ (if --ui)")
    typer.echo(f"Crawler: http://127.0.0.1:{port}/crawler")
    run_autonomous(
        data_dir=data_dir,
        queries_file=queries_file,
        patents_per_query=limit,
        parse_batch=parse_batch,
        collect_sleep=sleep,
        with_ui=with_ui,
        ui_port=port,
        model=model,
        api_key=api_key,
        crawler=crawler,
    )


@app.command("run-full-pipeline")
def run_full_pipeline(
    query: str,
    limit: int = 10,
    db_path: Optional[Path] = typer.Option(None, "--db"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    parser: str = typer.Option("gemini", help="Parser backend: gemini"),
    api_key: str | None = typer.Option(None, help="Google Gemini API key"),
    model: str = "gemini-2.0-flash",
) -> None:
    """Collect US patent pages by query, parse them, and populate the review database."""
    db = _open_db(db_path, data_dir)
    collect_pipeline = IngestionPipeline(database=db, parser=RawDocumentParserStub())
    urls = collect_pipeline.search_google_patents(query=query, limit=limit)
    if not urls:
        db.close()
        typer.echo(f"No patent URLs found for query: {query}")
        raise typer.Exit(code=1)
    for url in urls:
        collect_pipeline.collect_url(url)
    db.close()

    db = _open_db(db_path, data_dir)
    if parser.lower() != "gemini":
        raise typer.BadParameter("Only the Gemini parser is supported now.")
    parse_pipeline = IngestionPipeline(database=db, parser=GeminiLLMParser(api_key=api_key, model=model))
    records = parse_pipeline.parse_all()
    db.close()
    typer.echo(f"Collected and parsed {len(records)} patent records from {len(urls)} URLs")


@app.command("download-bulk")
def download_bulk(
    year: int = typer.Argument(2024, help="USPTO grant year"),
    week: int = typer.Argument(1, help="ISO week number (1–53)"),
    limit: int = typer.Option(500, help="Max patents to download"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="CSV output path (default: patents_YYYY_wWW.csv)"),
    chemistry_only: bool = typer.Option(True, "--chemistry/--all", help="Filter to CPC section C (Chemistry)"),
) -> None:
    """Download weekly bulk patent metadata from PatentsView as a CSV."""
    from .bulk_downloader import get_bulk_patent_data

    out_path = output or Path(f"patents_{year}_w{week:02d}.csv")
    typer.echo(f"Downloading year={year} week={week} (chemistry_only={chemistry_only}, limit={limit}) …")
    records = get_bulk_patent_data(year=year, week=week, limit=limit, output_csv=out_path, chemistry_only=chemistry_only)
    typer.echo(f"Downloaded {len(records)} patents → {out_path}")


@app.command("ingest-csv")
def ingest_csv_cmd(
    csv_file: Path = typer.Argument(..., help="CSV file to ingest (patent_id, title, abstract, raw_text, url columns)"),
    db_path: Optional[Path] = typer.Option(None, "--db"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    no_enqueue: bool = typer.Option(False, "--no-enqueue", help="Store documents without queuing for parsing"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--allow-duplicates"),
) -> None:
    """Ingest a bulk patent CSV into the database and queue rows for LLM parsing.

    Compatible with CSVs produced by 'download-bulk' or any CSV with columns:
    patent_id, title, abstract, raw_text (and optionally: date, url).

    After ingesting, run 'run-worker' to parse the queued documents with Gemini.
    """
    from .csv_ingestion import ingest_csv_file

    if not csv_file.exists():
        typer.echo(f"File not found: {csv_file}", err=True)
        raise typer.Exit(code=1)

    db = _open_db(db_path, data_dir)
    n = ingest_csv_file(csv_file, db, enqueue=not no_enqueue, skip_existing=skip_existing, verbose=True)
    db.close()
    typer.echo(f"Ingested {n} documents from {csv_file}")
    if not no_enqueue:
        typer.echo("Run 'run-worker' to parse the queued documents with Gemini.")


@app.command("ingest-ord")
def ingest_ord_cmd(
    limit: int = typer.Option(5000, help="Max reactions to import"),
    n_files: int = typer.Option(10, help="Number of ORD .pb.gz files to download/process"),
    pb_dir: Path = typer.Option(Path("ord_data"), "--pb-dir", help="Directory for .pb.gz files"),
    db_path: Optional[Path] = typer.Option(None, "--db"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir"),
    no_download: bool = typer.Option(False, "--no-download", help="Skip downloading; use existing files in --pb-dir"),
) -> None:
    """Import reactions from the Open Reaction Database (ORD) via GitHub LFS.

    Downloads .pb.gz files from the ORD GitHub repository and inserts reactions
    directly — no LLM needed (data is expert-curated).

    Requires: pip install ord-schema
    """
    from .ord_pb_ingestion import ingest_ord_pb_files

    db = _open_db(db_path, data_dir)
    try:
        n = ingest_ord_pb_files(
            db,
            pb_dir=pb_dir,
            n_files=n_files,
            limit=limit,
            download=not no_download,
            verbose=True,
        )
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        db.close()
        raise typer.Exit(code=1)
    db.close()
    typer.echo(f"ORD import complete: {n} reactions added.")


if __name__ == "__main__":
    app()
