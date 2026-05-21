"""Run the Gemini LLM worker on all pending queue items.

Usage:
    python run_worker.py           # parse all pending
    python run_worker.py --limit 10  # parse at most 10
"""
import argparse
import json
import os
import time

os.environ.setdefault("PATENT_DATA_DIR", "./data")

from src.patent_pipeline.database import PatentDatabase
from src.patent_pipeline.llm_parser import GeminiLLMParser
from src.patent_pipeline.models import RawDocument
from datetime import datetime, timezone


def run_worker(db: PatentDatabase, parser: GeminiLLMParser, limit: int = 1000) -> dict:
    stats = {"parsed": 0, "reactions": 0, "errors": 0}
    while stats["parsed"] < limit:
        job = db.get_next_queue_item()
        if job is None:
            print(f"\n[WORKER] Queue empty. Done.")
            break
        queue_id = job["id"]
        raw_doc_id = job["raw_document_id"]
        row = db.connection.execute(
            "SELECT * FROM raw_documents WHERE id=?", (raw_doc_id,)
        ).fetchone()
        if not row:
            db.update_queue_status(queue_id, "error")
            continue
        title = str(row["title"] or row["source_url"] or "")[:60].encode("ascii", "replace").decode()
        print(f"[{stats['parsed']+1}] {title}")
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
        retries = 0
        success = False
        while retries < 4:
            try:
                patent = parser.parse(doc)
                db.upsert_patent(patent)
                rxn_count = 0
                for rxn in patent.reactions:
                    rxn.metadata["source"] = "patent"
                    try:
                        db._insert_reaction(rxn)
                        db.connection.execute(
                            "UPDATE reactions SET source='patent' WHERE reaction_id=?",
                            (rxn.reaction_id,),
                        )
                        rxn_count += 1
                    except Exception:
                        pass
                db.connection.commit()
                db.update_queue_status(queue_id, "done")
                stats["parsed"] += 1
                stats["reactions"] += rxn_count
                conf_str = str([f"{r.metadata.get('confidence', 0):.2f}" for r in patent.reactions])
                print(f"  {rxn_count} rxns, conf={conf_str}")
                success = True
                break
            except Exception as exc:
                err = str(exc)
                if "429" in err:
                    wait = 30 * (retries + 1)
                    print(f"  RATE LIMIT - waiting {wait}s (retry {retries+1}/4)...")
                    time.sleep(wait)
                    retries += 1
                else:
                    print(f"  ERROR: {err[:150]}")
                    db.update_queue_status(queue_id, "error")
                    stats["errors"] += 1
                    break
        if not success and retries >= 4:
            db.update_queue_status(queue_id, "error")
            stats["errors"] += 1
        time.sleep(5)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()

    db = PatentDatabase()
    pending = db.connection.execute(
        "SELECT COUNT(*) FROM parse_queue WHERE status='pending'"
    ).fetchone()[0]
    print(f"DB: {db.db_path}")
    print(f"Pending documents: {pending}")
    try:
        parser = GeminiLLMParser()
        print(f"Model: {parser.model}")
    except ValueError as e:
        print(f"ERROR: {e}")
        db.close()
        return

    print("-" * 50)
    stats = run_worker(db, parser, limit=args.limit)
    print("-" * 50)

    total = db.connection.execute("SELECT COUNT(*) FROM reactions").fetchone()[0]
    patent_rxns = db.connection.execute(
        "SELECT COUNT(*) FROM reactions WHERE source='patent'"
    ).fetchone()[0]
    print(f"Parsed: {stats['parsed']} docs | Reactions extracted: {stats['reactions']} | Errors: {stats['errors']}")
    print(f"DB totals: {total} reactions ({patent_rxns} LLM-extracted, {total-patent_rxns} ORD)")
    db.close()


if __name__ == "__main__":
    main()
