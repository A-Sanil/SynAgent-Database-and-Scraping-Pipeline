"""Auto-drain the parse queue by running drain_queue.py as a subprocess."""
import subprocess
import sys
import os
import time

os.environ.setdefault("PATENT_DATA_DIR", "./data")

from src.patent_pipeline.database import PatentDatabase

max_iters = 60
python = sys.executable

for i in range(max_iters):
    db = PatentDatabase()
    pending = db.connection.execute(
        "SELECT COUNT(*) FROM parse_queue WHERE status='pending'"
    ).fetchone()[0]
    done = db.connection.execute(
        "SELECT COUNT(*) FROM parse_queue WHERE status='done'"
    ).fetchone()[0]
    rxns = db.connection.execute(
        "SELECT COUNT(*) FROM reactions WHERE source='patent'"
    ).fetchone()[0]
    db.close()

    print(f"[iter {i+1}] pending={pending} done={done} LLM-rxns={rxns}")
    sys.stdout.flush()

    if pending == 0:
        print("Queue empty! Done.")
        break

    # Reset stuck and run drain_queue
    subprocess.run([python, "reset_queue.py"], capture_output=True)
    result = subprocess.run(
        [python, "-u", "drain_queue.py"],
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if result.stdout:
        lines = result.stdout.strip().splitlines()
        for line in lines[-5:]:  # last 5 lines
            print(" ", line)
    sys.stdout.flush()
    time.sleep(2)

db = PatentDatabase()
total = db.connection.execute("SELECT COUNT(*) FROM reactions").fetchone()[0]
patent_r = db.connection.execute(
    "SELECT COUNT(*) FROM reactions WHERE source='patent'"
).fetchone()[0]
done_final = db.connection.execute(
    "SELECT COUNT(*) FROM parse_queue WHERE status='done'"
).fetchone()[0]
print(f"\n=== Final: {done_final} papers parsed, {patent_r} LLM reactions, {total} total reactions ===")
db.close()
