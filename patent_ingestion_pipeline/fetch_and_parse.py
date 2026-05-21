"""Fetch chemistry synthesis papers from Europe PMC and run Gemini LLM parsing.

Usage:
    python fetch_and_parse.py            # fetch 30 papers + parse all pending
    python fetch_and_parse.py --fetch-only   # only download, don't parse
    python fetch_and_parse.py --parse-only   # only parse what's already queued
    python fetch_and_parse.py --limit 10     # parse at most 10 documents
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

# Force UTF-8 on Windows so Unicode chars in paper titles don't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8:replace")

os.environ.setdefault("PATENT_DATA_DIR", "./data")

from src.patent_pipeline.database import PatentDatabase
from src.patent_pipeline.models import RawDocument
from src.patent_pipeline.llm_parser import GeminiLLMParser

# ---------------------------------------------------------------------------
# Europe PMC queries that reliably return reaction-rich abstracts
# ---------------------------------------------------------------------------
QUERIES = [
    "synthesis yield reaction temperature solvent",
    "organic synthesis catalyst yield",
    "total synthesis natural product",
    "palladium coupling reaction yield",
    "asymmetric synthesis enantioselective",
]

EPMC_URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    "?query={q}&resultType=core&pageSize=25&cursorMark={cursor}&format=json"
)
PAGES_PER_QUERY = 4  # up to 100 results per query


def fetch_epmc_papers(n_per_query: int = 25) -> list[dict]:
    papers: list[dict] = []
    seen_ids: set[str] = set()
    for q in QUERIES:
        cursor = "*"
        for page in range(PAGES_PER_QUERY):
            url = EPMC_URL.format(q=urllib.parse.quote_plus(q), cursor=urllib.parse.quote_plus(cursor))
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "SynAgent/1.0"})
                with urllib.request.urlopen(req, timeout=15) as r:
                    data = json.loads(r.read())
                results = data.get("resultList", {}).get("result", [])
                if not results:
                    break
                for item in results:
                    pid = item.get("id") or item.get("pmid") or ""
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)
                    title = item.get("title", "")
                    abstract = item.get("abstractText", "")
                    if not abstract:
                        continue
                    source_url = (
                        item.get("doi") and f"https://doi.org/{item['doi']}"
                    ) or f"https://europepmc.org/article/MED/{pid}"
                    papers.append({
                        "id": pid,
                        "title": title,
                        "abstract": abstract,
                        "source_url": source_url,
                        "source_type": "europepmc",
                        "journal": item.get("journalTitle", ""),
                        "date": item.get("firstPublicationDate", ""),
                    })
                cursor = data.get("nextCursorMark", "")
                if not cursor:
                    break
            except Exception as exc:
                print(f"  [WARN] Query {q!r} page {page+1} failed: {exc}")
                break
            time.sleep(0.3)
        print(f"  Fetched from {q!r}: {len(papers)} total so far")
        time.sleep(0.3)
    return papers


def enqueue_papers(db: PatentDatabase, papers: list[dict]) -> int:
    import urllib.parse
    inserted = 0
    for p in papers:
        if db.has_source_url(p["source_url"]):
            print(f"  [SKIP] already in DB: {p['source_url']}")
            continue
        raw_text = f"{p['title']}\n\n{p['abstract']}"
        if p.get("journal"):
            raw_text += f"\n\nJournal: {p['journal']}"
        doc = RawDocument(
            source_url=p["source_url"],
            source_type=p["source_type"],
            title=p["title"] or None,
            fetched_at=datetime.now(tz=timezone.utc),
            content_type="text/plain",
            raw_text=raw_text,
            raw_html=None,
            metadata={
                "epmc_id": p["id"],
                "journal": p["journal"],
                "date": p["date"],
                "ingest_source": "europepmc",
            },
        )
        db.add_raw_document(doc)
        row = db.connection.execute(
            "SELECT id FROM raw_documents WHERE source_url = ? ORDER BY id DESC LIMIT 1",
            (p["source_url"],),
        ).fetchone()
        if row:
            db.enqueue_raw_document(int(row["id"]))
            inserted += 1
    return inserted


def run_worker(db: PatentDatabase, parser: GeminiLLMParser, limit: int = 100) -> int:
    parsed = 0
    while parsed < limit:
        job = db.get_next_queue_item()
        if job is None:
            print(f"\n[WORKER] Queue empty after {parsed} documents.")
            break
        queue_id, raw_doc_id = job["id"], job["raw_document_id"]
        row = db.connection.execute(
            "SELECT * FROM raw_documents WHERE id = ?", (raw_doc_id,)
        ).fetchone()
        if row is None:
            db.update_queue_status(queue_id, "error")
            continue

        doc = RawDocument(
            source_url=row["source_url"],
            source_type=row["source_type"],
            title=row["title"],
            fetched_at=datetime.now(tz=timezone.utc),
            content_type=row["content_type"],
            raw_text=row["raw_text"],
            raw_html=row["raw_html"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )
        raw_title = (doc.title or doc.source_url or "")[:80]
        title_safe = raw_title.encode("ascii", "replace").decode("ascii")
        print(f"\n[WORKER] Parsing ({parsed+1}): {title_safe}")
        retries = 0
        success = False
        while retries < 3:
            try:
                patent = parser.parse(doc)
                db.upsert_patent(patent)
                for rxn in patent.reactions:
                    rxn.metadata["source"] = "patent"
                    try:
                        db._insert_reaction(rxn)
                        db.connection.execute(
                            "UPDATE reactions SET source='patent' WHERE reaction_id=?",
                            (rxn.reaction_id,),
                        )
                    except Exception:
                        pass
                db.update_queue_status(queue_id, "done")
                n_rxns = len(patent.reactions)
                conf_list = [f"{r.metadata.get('confidence', 0):.2f}" for r in patent.reactions]
                print(f"  OK: {n_rxns} reactions, conf={conf_list}")
                parsed += 1
                success = True
                break
            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str:
                    wait = 15 * (retries + 1)
                    print(f"  [RATE LIMIT] waiting {wait}s ...")
                    time.sleep(wait)
                    retries += 1
                else:
                    print(f"  [ERROR] {err_str[:200]}")
                    db.update_queue_status(queue_id, "error")
                    break
        if not success and retries >= 3:
            print("  [FAIL] Too many rate limit retries, skipping.")
            db.update_queue_status(queue_id, "error")
        time.sleep(5)  # 5s between calls = max 12 RPM (Gemini free: 15 RPM)
    return parsed


def main():
    import argparse
    import urllib.parse  # ensure available

    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--parse-only", action="store_true")
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    db = PatentDatabase()
    print(f"DB: {db.db_path}")

    if not args.parse_only:
        print("\n[FETCH] Querying Europe PMC for chemistry papers ...")
        papers = fetch_epmc_papers()
        print(f"\n[FETCH] Found {len(papers)} unique papers. Enqueueing ...")
        n = enqueue_papers(db, papers)
        print(f"[FETCH] {n} new documents added to parse queue.")

    if args.fetch_only:
        total = db.connection.execute(
            "SELECT COUNT(*) FROM parse_queue WHERE status='pending'"
        ).fetchone()[0]
        print(f"\n[INFO] {total} documents pending in queue. Run again without --fetch-only to parse.")
        db.close()
        return

    pending = db.connection.execute(
        "SELECT COUNT(*) FROM parse_queue WHERE status='pending'"
    ).fetchone()[0]
    print(f"\n[WORKER] {pending} documents pending. Starting Gemini parser ...")

    try:
        parser = GeminiLLMParser()
    except ValueError as e:
        print(f"[ERROR] {e}\nSet GEMINI_API_KEY in .env or environment.")
        db.close()
        sys.exit(1)

    n = run_worker(db, parser, limit=args.limit)
    total_reactions = db.connection.execute("SELECT COUNT(*) FROM reactions").fetchone()[0]
    patent_reactions = db.connection.execute(
        "SELECT COUNT(*) FROM reactions WHERE source='patent'"
    ).fetchone()[0]
    print(f"\n[DONE] Parsed {n} documents.")
    print(f"       Reactions in DB: {total_reactions:,} total ({patent_reactions} from LLM, {total_reactions-patent_reactions} from ORD)")
    db.close()


if __name__ == "__main__":
    import urllib.parse
    main()
