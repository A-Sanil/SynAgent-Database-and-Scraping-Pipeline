"""
Pre-calculates fingerprints for every reaction in the database.

Phase 1 (already done): Morgan ECFP4 per product + reactant → product_fp, reactant_fp
Phase 2 (this run):     DRFP per full reaction SMARTS      → reaction_fp

DRFP (Differential Reaction Fingerprint) captures the actual chemical transformation
— what bonds broke and formed — making it far better for reaction yield prediction
than comparing molecules individually.

Usage:
    python "Agent tools/precalc_fingerprints.py"
"""

import argparse
import sqlite3
import numpy as np
from drfp import DrfpEncoder
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

DB_PATH   = r"D:\SynAgent\db\patent_pipeline.db"
FP_SIZE   = 2048
N_WORKERS = max(1, cpu_count() - 1)   # leave 1 core free for the OS
CHUNK     = 256    # reactions per worker task — small enough to keep all cores busy
DB_COMMIT = 5000   # write to DB every N completions


def _encode_chunk(args: tuple) -> list[tuple[str, bytes | None]]:
    """Worker function: encode a chunk of (id, smarts) pairs.
    Returns list of (reaction_id, blob_or_None).
    Runs in a subprocess so imports DrfpEncoder fresh — no GIL contention.
    """
    from drfp import DrfpEncoder
    import numpy as np
    pairs = []
    ids, smarts_list = args
    try:
        # Vectorised encode for the whole chunk at once
        fps = DrfpEncoder.encode(smarts_list, n_folded_length=FP_SIZE)
        for rid, fp in zip(ids, fps):
            pairs.append((rid, np.packbits(fp.astype(np.uint8)).tobytes()))
    except Exception:
        # Fall back one-by-one if any SMARTS is malformed
        for rid, smarts in zip(ids, smarts_list):
            try:
                fp = DrfpEncoder.encode([smarts], n_folded_length=FP_SIZE)[0].astype(np.uint8)
                pairs.append((rid, np.packbits(fp).tobytes()))
            except Exception:
                pairs.append((rid, None))
    return pairs


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",       default=DB_PATH, help="Path to SQLite database")
    parser.add_argument("--workers",  type=int, default=N_WORKERS,
                        help="Number of parallel worker processes")
    parser.add_argument("--chunk",    type=int, default=CHUNK,
                        help="Reactions per worker task")
    args, _ = parser.parse_known_args()

    conn = sqlite3.connect(args.db, timeout=60)
    print(f"DB:      {args.db}")
    print(f"Workers: {args.workers}  Chunk: {args.chunk}  Commit every: {DB_COMMIT}")
    cur  = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")

    existing = {row[1] for row in cur.execute("PRAGMA table_info(reactions)")}
    if "reaction_fp" not in existing:
        cur.execute("ALTER TABLE reactions ADD COLUMN reaction_fp BLOB")
        conn.commit()
        print("Added column: reaction_fp")

    cur.execute(
        "SELECT reaction_id, reaction_smarts FROM reactions "
        "WHERE reaction_smarts IS NOT NULL "
        "  AND reaction_smarts LIKE '%>>%' "
        "  AND reaction_fp IS NULL"
    )
    rows  = cur.fetchall()
    total = len(rows)
    print(f"Reactions to fingerprint: {total:,}\n")

    if total == 0:
        print("Nothing to do — all reactions already have reaction_fp.")
        conn.close()
        return

    # Split into chunks for the worker pool
    chunks = [
        ([r[0] for r in rows[i:i+args.chunk]],
         [r[1] for r in rows[i:i+args.chunk]])
        for i in range(0, total, args.chunk)
    ]

    updated = skipped = 0
    pending_pairs = []   # buffer before DB write

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_encode_chunk, c): c for c in chunks}
        for i, future in enumerate(as_completed(futures), 1):
            for rid, blob in future.result():
                if blob is None:
                    skipped += 1
                else:
                    pending_pairs.append((blob, rid))
                    updated += 1

            # Flush to DB every DB_COMMIT reactions
            if len(pending_pairs) >= DB_COMMIT or i == len(futures):
                if pending_pairs:
                    cur.executemany(
                        "UPDATE reactions SET reaction_fp = ? WHERE reaction_id = ?",
                        pending_pairs,
                    )
                    conn.commit()
                    pending_pairs = []
                done_rxn = (updated + skipped)
                pct = done_rxn / total * 100
                print(f"  {done_rxn:,}/{total:,} ({pct:.1f}%) "
                      f"— updated {updated:,}  skipped {skipped:,}")

    conn.close()
    print(f"\nDone. {updated:,} reaction_fp blobs written, {skipped:,} skipped.")


if __name__ == "__main__":
    run()
