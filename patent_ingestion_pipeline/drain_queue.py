"""Drain the parse queue completely: keep running the worker until queue is empty.

This script loops, running batches of 5, until done or all errors.
"""
import os, sys, time

# Suppress RDKit/chem stderr noise so it doesn't confuse shell redirection
sys.stderr = open(os.devnull, "w")

os.environ.setdefault("PATENT_DATA_DIR", "./data")

from src.patent_pipeline.database import PatentDatabase
from src.patent_pipeline.llm_parser import GeminiLLMParser
from src.patent_pipeline.models import RawDocument
from datetime import datetime, timezone
import json

db = PatentDatabase()
try:
    parser = GeminiLLMParser()
except ValueError as e:
    print(f"ERROR: {e}")
    db.close()
    sys.exit(1)

print(f"DB: {db.db_path}")
print(f"Model: {parser.model}")
total_parsed = 0
total_reactions = 0

while True:
    # Reset any stuck "running" items
    n_reset = db.connection.execute(
        "UPDATE parse_queue SET status='pending' WHERE status='running'"
    ).rowcount
    db.connection.commit()
    if n_reset:
        print(f"[+] Reset {n_reset} stuck running items")

    pending = db.connection.execute(
        "SELECT COUNT(*) FROM parse_queue WHERE status='pending'"
    ).fetchone()[0]
    done = db.connection.execute(
        "SELECT COUNT(*) FROM parse_queue WHERE status='done'"
    ).fetchone()[0]
    errors = db.connection.execute(
        "SELECT COUNT(*) FROM parse_queue WHERE status='error'"
    ).fetchone()[0]
    total_rxns = db.connection.execute("SELECT COUNT(*) FROM reactions").fetchone()[0]
    patent_rxns = db.connection.execute(
        "SELECT COUNT(*) FROM reactions WHERE source='patent'"
    ).fetchone()[0]
    print(f"\n--- Queue: pending={pending} done={done} errors={errors} | Rxns: {total_rxns} ({patent_rxns} LLM) ---")

    if pending == 0:
        print("Queue empty! All done.")
        break

    job = db.get_next_queue_item()
    if job is None:
        print("No pending job found. Done.")
        break
    queue_id = job["id"]
    raw_doc_id = job["raw_document_id"]
    row = db.connection.execute("SELECT * FROM raw_documents WHERE id=?", (raw_doc_id,)).fetchone()
    if not row:
        db.update_queue_status(queue_id, "error")
        continue

    title = str(row["title"] or row["source_url"] or "")[:60].encode("ascii", "replace").decode()
    print(f"Parsing: {title}")

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
    while retries < 4:
        try:
            patent = parser.parse(doc)
            # Set source before upsert so it's included in the metadata
            for rxn in patent.reactions:
                rxn.metadata["source"] = "patent"
            # Pre-delete reactions by reaction_id to avoid UNIQUE constraint
            # (LLM may generate different patent_id between runs, so DELETE by patent_id alone may miss them)
            for rxn in patent.reactions:
                try:
                    db.connection.execute("DELETE FROM reactions WHERE reaction_id=?", (rxn.reaction_id,))
                except Exception:
                    pass
            db.connection.commit()
            db.upsert_patent(patent)  # handles DELETE by patent_id + INSERT for all reactions
            rxn_count = len(patent.reactions)
            # Set source column directly (upsert_patent uses INSERT without source col)
            for rxn in patent.reactions:
                try:
                    db.connection.execute(
                        "UPDATE reactions SET source='patent' WHERE reaction_id=?",
                        (rxn.reaction_id,),
                    )
                except Exception:
                    pass
            db.connection.commit()
            db.update_queue_status(queue_id, "done")
            total_parsed += 1
            total_reactions += rxn_count
            conf_str = str([f"{r.metadata.get('confidence', 0):.2f}" for r in patent.reactions])
            print(f"  OK: {rxn_count} rxns, conf={conf_str}")
            break
        except Exception as exc:
            err = str(exc)
            if "429" in err:
                wait = 30 * (retries + 1)
                print(f"  RATE LIMIT - retry {retries+1}, waiting {wait}s", flush=True)
                for _ in range(wait):
                    time.sleep(1)
                    sys.stdout.write(".")
                    sys.stdout.flush()
                print(flush=True)
                retries += 1
            else:
                print(f"  ERROR: {err[:120]}")
                db.update_queue_status(queue_id, "error")
                break
    if retries >= 4:
        db.update_queue_status(queue_id, "error")

    # Paced delay between requests — 8s = ~7 RPM, well under 15 RPM free limit
    # Heartbeat dots keep the task runner from killing idle processes
    for _ in range(8):
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\n")
    sys.stdout.flush()

total = db.connection.execute("SELECT COUNT(*) FROM reactions").fetchone()[0]
patent_r = db.connection.execute("SELECT COUNT(*) FROM reactions WHERE source='patent'").fetchone()[0]
print(f"\n=== COMPLETE ===")
print(f"Parsed: {total_parsed} docs | Reactions: {total_reactions} extracted")
print(f"DB: {total} total reactions ({patent_r} LLM, {total-patent_r} ORD)")
db.close()
