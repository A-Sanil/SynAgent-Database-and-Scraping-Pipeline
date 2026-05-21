"""Fetch a broader set of chemistry synthesis papers from Europe PMC.

Uses 15 diverse queries targeting reaction-rich abstracts to populate the parse queue.
Skips papers already in DB (deduped by source_url).
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

os.environ.setdefault("PATENT_DATA_DIR", "./data")

from src.patent_pipeline.database import PatentDatabase
from src.patent_pipeline.models import RawDocument

QUERIES = [
    # Specific reaction conditions / reagents
    "Grignard reaction organomagnesium synthesis",
    "lithiation deprotonation organolithium synthesis",
    "oxidation alcohol ketone aldehyde synthesis",
    "reduction carbonyl borohydride synthesis",
    "epoxidation asymmetric Sharpless synthesis",
    "click chemistry azide alkyne cycloaddition",
    "Wittig reaction phosphorus ylide synthesis",
    "Mannich reaction synthesis amino carbonyl",
    "aldol reaction condensation synthesis",
    "Mitsunobu reaction inversion stereochemistry",
    # Material / drug classes
    "heterocycle indole synthesis yield conditions",
    "beta-lactam antibiotic synthesis",
    "macrolide synthesis ring-closing",
    "alkaloid synthesis total yield",
    "terpenoid synthesis cyclization conditions",
    "PROTAC synthesis linker yield",
    "nucleoside analog synthesis glycosylation",
    "fluorous synthesis conditions solvent",
    "electrochemical synthesis oxidation reduction",
    "flow chemistry continuous synthesis yield",
]

EPMC_URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    "?query={q}&resultType=core&pageSize=25&cursorMark={cursor}&format=json"
)
PAGES_PER_QUERY = 3  # fetch up to 3 pages (75 results) per query


def fetch_papers() -> list[dict]:
    papers: list[dict] = []
    seen_ids: set[str] = set()
    for i, q in enumerate(QUERIES):
        new_this_query = 0
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
                    abstract = item.get("abstractText", "")
                    if not abstract:
                        continue
                    seen_ids.add(pid)
                    title = item.get("title", "")
                    doi = item.get("doi", "")
                    source_url = (
                        f"https://doi.org/{doi}" if doi else f"https://europepmc.org/article/MED/{pid}"
                    )
                    papers.append({
                        "id": pid,
                        "title": title,
                        "abstract": abstract,
                        "source_url": source_url,
                        "source_type": "europepmc",
                        "journal": item.get("journalTitle", ""),
                        "date": item.get("firstPublicationDate", ""),
                    })
                    new_this_query += 1
                # get cursor for next page
                cursor = data.get("nextCursorMark", "")
                if not cursor or cursor == data.get("request", {}).get("cursorMark"):
                    break
            except Exception as exc:
                print(f"  [WARN] Query {i+1} page {page+1} failed: {exc}")
                break
            time.sleep(0.3)
        print(f"  Query {i+1:>2}/{len(QUERIES)}: {q[:50]!r} -> {new_this_query} new papers")
        time.sleep(0.3)
    print(f"\nTotal unique papers fetched: {len(papers)}")
    return papers


def enqueue_papers(db: PatentDatabase, papers: list[dict]) -> int:
    inserted = 0
    skipped = 0
    for p in papers:
        if db.has_source_url(p["source_url"]):
            skipped += 1
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
    print(f"Enqueued {inserted} new documents ({skipped} already in DB, skipped)")
    return inserted


if __name__ == "__main__":
    db = PatentDatabase()
    print(f"DB: {db.db_path}")
    print("\nFetching papers from Europe PMC ...")
    papers = fetch_papers()
    print("\nEnqueueing new papers ...")
    n = enqueue_papers(db, papers)

    pending = db.connection.execute(
        "SELECT COUNT(*) FROM parse_queue WHERE status='pending'"
    ).fetchone()[0]
    total_docs = db.connection.execute("SELECT COUNT(*) FROM raw_documents").fetchone()[0]
    print(f"\nQueue now has {pending} pending documents to parse.")
    print(f"Total raw documents in DB: {total_docs}")
    db.close()
