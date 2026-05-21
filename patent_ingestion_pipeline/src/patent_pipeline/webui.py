from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import get_db_path, resolve_data_dir
from .crawl_tracker import tracker
from .crawler_registry import PROFILES
from .database import PatentDatabase

load_dotenv()

app = FastAPI(title="Patent Parser Review UI")


def _resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


BASE_DIR = _resource_root()
_templates_dir = BASE_DIR / "templates"
_static_dir = BASE_DIR / "static"
if not _templates_dir.is_dir():
    _templates_dir = Path(__file__).resolve().parent / "templates"
    _static_dir = Path(__file__).resolve().parent / "static"

templates = Jinja2Templates(directory=str(_templates_dir))
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


def _db() -> PatentDatabase:
    return PatentDatabase()


def _ctx(extra: dict | None = None) -> dict:
    base = resolve_data_dir()
    payload = {
        "data_dir": str(base),
        "db_path": str(get_db_path(base)),
    }
    if extra:
        payload.update(extra)
    return payload


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    db = _db()
    rows = db.connection.execute(
        "SELECT patent_id, title, reviewed FROM patents ORDER BY reviewed, publication_date DESC LIMIT 200"
    ).fetchall()
    db.close()
    return templates.TemplateResponse(
        request,
        "index.html",
        _ctx({"patents": rows}),
    )


@app.post("/enqueue_all")
def web_enqueue_all():
    db = _db()
    cur = db.connection.execute("SELECT id FROM raw_documents ORDER BY id")
    count = 0
    for r in cur.fetchall():
        try:
            db.enqueue_raw_document(int(r["id"]))
            count += 1
        except Exception:
            pass
    db.close()
    return RedirectResponse(url="/", status_code=303)


@app.post("/upload_patent")
async def upload_patent(file: UploadFile = File(...)):
    import uuid
    import csv
    from io import StringIO
    from datetime import datetime, timezone
    from .models import RawDocument

    db = _db()
    content = await file.read()
    try:
        text_content = content.decode("utf-8")
    except UnicodeDecodeError:
        text_content = content.decode("latin-1")

    filename = (file.filename or "uploaded.txt").lower()

    if filename.endswith(".csv"):
        reader = csv.DictReader(StringIO(text_content))
        for row in reader:
            doc = RawDocument(
                source_url=row.get("url") or row.get("source_url") or f"upload://{uuid.uuid4().hex[:8]}",
                source_type="bulk_csv",
                fetched_at=datetime.now(timezone.utc),
                title=row.get("title") or row.get("patent_title") or "Untitled",
                content_type="text/plain",
                raw_text=row.get("text") or row.get("description") or row.get("abstract") or "",
                metadata={"fetcher": "manual_upload"}
            )
            db.add_raw_document(doc)
            db_row = db.connection.execute("SELECT id FROM raw_documents ORDER BY id DESC LIMIT 1").fetchone()
            if db_row:
                db.enqueue_raw_document(int(db_row["id"]))
    elif filename.endswith(".jsonl"):
        for line in text_content.strip().split("\n"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                doc = RawDocument(
                    source_url=row.get("url") or row.get("source_url") or f"upload://{uuid.uuid4().hex[:8]}",
                    source_type="bulk_jsonl",
                    fetched_at=datetime.now(timezone.utc),
                    title=row.get("title") or row.get("patent_title") or "Untitled",
                    content_type="text/plain",
                    raw_text=row.get("text") or row.get("description") or row.get("abstract") or "",
                    metadata={"fetcher": "manual_upload"}
                )
                db.add_raw_document(doc)
                db_row = db.connection.execute("SELECT id FROM raw_documents ORDER BY id DESC LIMIT 1").fetchone()
                if db_row:
                    db.enqueue_raw_document(int(db_row["id"]))
            except json.JSONDecodeError:
                pass
    else:
        source_type = "patent_html" if filename.endswith(".html") else "patent_txt"
        doc = RawDocument(
            source_url=f"upload://{filename}_{uuid.uuid4().hex[:8]}",
            source_type=source_type,
            fetched_at=datetime.now(timezone.utc),
            title=filename,
            content_type="text/html" if source_type == "patent_html" else "text/plain",
            raw_text=text_content,
            raw_html=text_content if source_type == "patent_html" else None,
            metadata={"fetcher": "manual_upload"}
        )
        db.add_raw_document(doc)
        row = db.connection.execute("SELECT id FROM raw_documents ORDER BY id DESC LIMIT 1").fetchone()
        if row:
            db.enqueue_raw_document(int(row["id"]))
            
    db.close()
    return RedirectResponse(url="/", status_code=303)



@app.get("/search", response_class=HTMLResponse)
def web_search(request: Request, q: str = ""):
    db = _db()
    patents = []
    reactions = []
    if q:
        try:
            res = db.search_text(q)
            patents = res.get("patents", [])
            reactions = res.get("reactions", [])
        except Exception:
            pass
    db.close()
    return templates.TemplateResponse(
        request,
        "search.html",
        _ctx({"q": q, "patents": patents, "reactions": reactions}),
    )


@app.get("/batch_review", response_class=HTMLResponse)
def batch_review(request: Request):
    db = _db()
    rows = db.list_low_confidence_reactions()
    db.close()
    return templates.TemplateResponse(
        request,
        "batch_review.html",
        _ctx({"reactions": rows}),
    )


@app.post("/api/active_learning")
def api_active_learning(payload: dict = Body(...)):
    rid = payload.get("reaction_id")
    pid = payload.get("patent_id")
    field = payload.get("field")
    old = payload.get("old_value")
    new = payload.get("new_value")
    user = payload.get("user") or "web"
    if not rid or not field:
        raise HTTPException(status_code=400, detail="reaction_id and field are required")

    db = _db()
    with db.connection:
        db.connection.execute(
            "INSERT INTO active_learning (reaction_id, patent_id, field, old_value, new_value, user) VALUES (?, ?, ?, ?, ?, ?)",
            (rid, pid, field, old, new, user),
        )
        if new is not None:
            if field == "product_smiles":
                from .chem_ner import normalize_smiles

                normalized = normalize_smiles(str(new))
                if normalized is None:
                    db.close()
                    raise HTTPException(status_code=400, detail="Invalid product SMILES")
                new = normalized
            db.update_reaction_field(str(rid), str(field), new)
            row = db.connection.execute("SELECT * FROM reactions WHERE reaction_id = ?", (rid,)).fetchone()
            if row:
                from .models import ReactionRecord

                rr = ReactionRecord(
                    reaction_id=row["reaction_id"],
                    patent_id=row["patent_id"],
                    reaction_smarts=row["reaction_smarts"],
                    reactant_smiles=json.loads(row["reactant_smiles_json"] or "[]"),
                    product_smiles=row["product_smiles"],
                    yield_percent=row["yield_percent"],
                    temperature_celsius=row["temperature_celsius"],
                    solvent=row["solvent"],
                    catalyst=row["catalyst"],
                    time_hours=row["time_hours"],
                    mechanism_text=row["mechanism_text"],
                    notes=row["notes"],
                )
                db._index_reaction_fts(rr)
    db.close()
    return {"status": "ok"}


@app.get("/patent/{patent_id}", response_class=HTMLResponse)
def view_patent(request: Request, patent_id: str):
    db = _db()
    p = db.connection.execute("SELECT * FROM patents WHERE patent_id = ?", (patent_id,)).fetchone()
    reactions = db.connection.execute("SELECT * FROM reactions WHERE patent_id = ?", (patent_id,)).fetchall()
    db.close()
    if p is None:
        raise HTTPException(status_code=404, detail="Patent not found")
    return templates.TemplateResponse(
        request,
        "patent.html",
        _ctx({"patent": p, "reactions": reactions}),
    )


@app.post("/patent/{patent_id}/approve")
def approve_patent(patent_id: str):
    db = _db()
    db.mark_patent_reviewed(patent_id, True)
    db.close()
    return RedirectResponse(url=f"/patent/{patent_id}", status_code=303)


@app.post("/patent/{patent_id}/update")
def update_patent(patent_id: str, title: str = Form(...), abstract: str = Form("")):
    db = _db()
    with db.connection:
        db.connection.execute(
            "UPDATE patents SET title = ?, abstract = ? WHERE patent_id = ?",
            (title, abstract, patent_id),
        )
        row = db.connection.execute("SELECT * FROM patents WHERE patent_id = ?", (patent_id,)).fetchone()
        if row:
            from .models import PatentRecord

            pr = PatentRecord(
                patent_id=row["patent_id"],
                title=row["title"],
                abstract=row["abstract"],
                source_url=row["source_url"],
                raw_text=row["raw_text"],
                reactions=[],
            )
            db._index_patent_fts(pr)
    db.close()
    return RedirectResponse(url=f"/patent/{patent_id}", status_code=303)


@app.post("/patent/{patent_id}/reaction/{reaction_id}/update")
def update_reaction(
    patent_id: str,
    reaction_id: str,
    product_smiles: str = Form(None),
    yield_percent: str = Form(None),
    notes: str = Form(None),
):
    db = _db()
    with db.connection:
        db.connection.execute(
            "UPDATE reactions SET product_smiles = ?, yield_percent = ?, notes = ? WHERE reaction_id = ?",
            (product_smiles, float(yield_percent) if yield_percent else None, notes, reaction_id),
        )
        row = db.connection.execute("SELECT * FROM reactions WHERE reaction_id = ?", (reaction_id,)).fetchone()
        if row:
            from .models import ReactionRecord

            rr = ReactionRecord(
                reaction_id=row["reaction_id"],
                patent_id=row["patent_id"],
                reaction_smarts=row["reaction_smarts"],
                reactant_smiles=json.loads(row["reactant_smiles_json"] or "[]"),
                product_smiles=row["product_smiles"],
                yield_percent=row["yield_percent"],
                temperature_celsius=row["temperature_celsius"],
                solvent=row["solvent"],
                catalyst=row["catalyst"],
                time_hours=row["time_hours"],
                mechanism_text=row["mechanism_text"],
                notes=row["notes"],
            )
            db._index_reaction_fts(rr)
    db.close()
    return RedirectResponse(url=f"/patent/{patent_id}", status_code=303)


@app.post("/api/reaction/update")
def api_update_reaction(payload: dict = Body(...)):
    patent_id = payload.get("patent_id")
    reaction_id = payload.get("reaction_id")
    product_smiles = payload.get("product_smiles")
    yp = payload.get("yield_percent")
    notes = payload.get("notes")
    from .chem_ner import normalize_smiles

    db = _db()
    if product_smiles:
        normalized = normalize_smiles(str(product_smiles))
        if normalized is None:
            db.close()
            raise HTTPException(status_code=400, detail="Invalid product SMILES")
        product_smiles = normalized
    with db.connection:
        db.connection.execute(
            "UPDATE reactions SET product_smiles = ?, yield_percent = ?, notes = ? WHERE reaction_id = ?",
            (product_smiles, float(yp) if yp not in (None, "") else None, notes, reaction_id),
        )
        row = db.connection.execute("SELECT * FROM reactions WHERE reaction_id = ?", (reaction_id,)).fetchone()
        if row:
            from .models import ReactionRecord

            rr = ReactionRecord(
                reaction_id=row["reaction_id"],
                patent_id=row["patent_id"],
                reaction_smarts=row["reaction_smarts"],
                reactant_smiles=json.loads(row["reactant_smiles_json"] or "[]"),
                product_smiles=row["product_smiles"],
                yield_percent=row["yield_percent"],
                temperature_celsius=row["temperature_celsius"],
                solvent=row["solvent"],
                catalyst=row["catalyst"],
                time_hours=row["time_hours"],
                mechanism_text=row["mechanism_text"],
                notes=row["notes"],
            )
            db._index_reaction_fts(rr)
    db.close()
    return {"status": "ok", "patent_id": patent_id, "reaction_id": reaction_id}


@app.get("/runworker", response_class=HTMLResponse)
def runworker_page(request: Request):
    """Legacy URL from the UI button — show worker controls."""
    return templates.TemplateResponse(
        request,
        "runworker.html",
        _ctx(),
    )


@app.post("/api/worker/tick")
def api_worker_tick():
    """Process one parse-queue job (same as a single worker iteration)."""
    from .llm_parser import GeminiLLMParser
    from .pipeline import IngestionPipeline

    db = _db()
    try:
        pipeline = IngestionPipeline(database=db, parser=GeminiLLMParser())
        processed = pipeline.process_queue_once()
    except ValueError as exc:
        db.close()
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    except Exception as exc:
        db.close()
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    db.close()
    return {
        "ok": True,
        "processed": processed,
        "message": "Parsed one document" if processed else "Queue empty — collect and enqueue patents first",
    }


@app.get("/status", response_class=HTMLResponse)
def status_page(request: Request):
    """Human-readable connection / database summary."""
    base = resolve_data_dir()
    db_path = get_db_path(base)
    payload = {
        "data_dir": str(base),
        "database": str(db_path),
        "database_exists": db_path.exists(),
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
        "counts": {"patents": 0, "reactions": 0, "raw_documents": 0, "queue_pending": 0},
    }
    if db_path.exists():
        db = PatentDatabase()
        payload["counts"]["patents"] = db.connection.execute("SELECT COUNT(*) FROM patents").fetchone()[0]
        payload["counts"]["reactions"] = db.connection.execute("SELECT COUNT(*) FROM reactions").fetchone()[0]
        payload["counts"]["raw_documents"] = db.connection.execute("SELECT COUNT(*) FROM raw_documents").fetchone()[0]
        payload["counts"]["queue_pending"] = db.connection.execute(
            "SELECT COUNT(*) FROM parse_queue WHERE status = 'pending'"
        ).fetchone()[0]
        db.close()
    return templates.TemplateResponse(request, "status.html", _ctx(payload))


@app.get("/crawler", response_class=HTMLResponse)
def crawler_dashboard(request: Request):
    return templates.TemplateResponse(request, "crawler.html", _ctx({"profiles": list(PROFILES.keys())}))


@app.get("/api/crawler/status")
def api_crawler_status():
    base = resolve_data_dir()
    db_path = get_db_path(base)
    payload: dict = {"crawler": tracker.snapshot(), "database": str(db_path), "counts": {}, "runs": []}
    if db_path.exists():
        db = PatentDatabase()
        payload["counts"] = {
            "patents": db.connection.execute("SELECT COUNT(*) FROM patents").fetchone()[0],
            "reactions": db.connection.execute("SELECT COUNT(*) FROM reactions").fetchone()[0],
            "raw_documents": db.connection.execute("SELECT COUNT(*) FROM raw_documents").fetchone()[0],
            "queue_pending": db.connection.execute(
                "SELECT COUNT(*) FROM parse_queue WHERE status = 'pending'"
            ).fetchone()[0],
        }
        runs = db.list_crawl_runs(limit=20)
        payload["runs"] = [dict(r) for r in runs]
        db.close()
    return JSONResponse(payload)


@app.get("/upload-csv", response_class=HTMLResponse)
def upload_csv_page(request: Request):
    """Form page for dropping a bulk patent CSV into the pipeline."""
    return templates.TemplateResponse(request, "upload_csv.html", _ctx())


@app.post("/api/upload-csv")
async def api_upload_csv(
    file: UploadFile = File(...),
    enqueue: bool = Form(True),
):
    """Accept a patent CSV upload, store each row as a RawDocument, and enqueue for parsing.

    Returns JSON with the number of documents ingested.
    Compatible CSV columns: patent_id, title, abstract, raw_text, url, date.
    """
    from .csv_ingestion import ingest_csv_bytes

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    db = _db()
    try:
        n = ingest_csv_bytes(
            content,
            filename=file.filename,
            db=db,
            enqueue=enqueue,
            skip_existing=True,
        )
    except Exception as exc:
        db.close()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    db.close()
    return JSONResponse({"ok": True, "ingested": n, "filename": file.filename})


@app.get("/api/status")
def api_status():
    base = resolve_data_dir()
    db_path = get_db_path(base)
    exists = db_path.exists()
    counts = {"patents": 0, "reactions": 0, "raw_documents": 0, "queue_pending": 0}
    if exists:
        db = PatentDatabase()
        counts["patents"] = db.connection.execute("SELECT COUNT(*) FROM patents").fetchone()[0]
        counts["reactions"] = db.connection.execute("SELECT COUNT(*) FROM reactions").fetchone()[0]
        counts["raw_documents"] = db.connection.execute("SELECT COUNT(*) FROM raw_documents").fetchone()[0]
        counts["queue_pending"] = db.connection.execute(
            "SELECT COUNT(*) FROM parse_queue WHERE status = 'pending'"
        ).fetchone()[0]
        db.close()
    return JSONResponse(
        {
            "data_dir": str(base),
            "database": str(db_path),
            "database_exists": exists,
            "counts": counts,
            "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
        }
    )
