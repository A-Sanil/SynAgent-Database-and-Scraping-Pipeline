"""Persistent loop: keeps restarting drain_queue.py until the queue is empty.

Handles rate-limit exhaustion by waiting before restarting.
Usage:
    python run_loop.py
    set PATENT_DATA_DIR=D:\\SynAgent && python run_loop.py
"""
import os
import sys
import subprocess
import time

os.environ.setdefault("PATENT_DATA_DIR", "./data")

from src.patent_pipeline.database import PatentDatabase

MAX_ROUNDS = 200
RESTART_WAIT = 60  # seconds to wait before restarting after failure

python = sys.executable

for round_num in range(1, MAX_ROUNDS + 1):
    # Check queue
    db = PatentDatabase()
    pending = db.connection.execute(
        "SELECT COUNT(*) FROM parse_queue WHERE status='pending'"
    ).fetchone()[0]
    done = db.connection.execute(
        "SELECT COUNT(*) FROM parse_queue WHERE status='done'"
    ).fetchone()[0]
    errors = db.connection.execute(
        "SELECT COUNT(*) FROM parse_queue WHERE status='error'"
    ).fetchone()[0]
    llm_rxns = db.connection.execute(
        "SELECT COUNT(*) FROM reactions WHERE source='patent'"
    ).fetchone()[0]
    db.close()

    print(f"\n[Round {round_num}] pending={pending} done={done} errors={errors} LLM-rxns={llm_rxns}",
          flush=True)

    if pending == 0:
        print("Queue empty — all done!", flush=True)
        break

    # Reset stuck/error items
    db = PatentDatabase()
    n_reset = db.connection.execute(
        "UPDATE parse_queue SET status='pending' WHERE status IN ('running','error')"
    ).rowcount
    db.connection.commit()
    db.close()
    if n_reset:
        print(f"  Reset {n_reset} stuck/error items", flush=True)

    # Run drain_queue as subprocess (inherits env, including PATENT_DATA_DIR)
    print(f"  Starting drain_queue.py ...", flush=True)
    result = subprocess.run(
        [python, "-u", "drain_queue.py"],
        text=True,
        timeout=7200,  # 2 hours max per drain run
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    if result.returncode == 0:
        print("  drain_queue finished cleanly.", flush=True)
    else:
        print(f"  drain_queue exited with code {result.returncode} — "
              f"waiting {RESTART_WAIT}s before retry ...", flush=True)
        for i in range(RESTART_WAIT):
            sys.stdout.write(".")
            sys.stdout.flush()
            time.sleep(1)
        print(flush=True)

print("\nrun_loop complete.", flush=True)
