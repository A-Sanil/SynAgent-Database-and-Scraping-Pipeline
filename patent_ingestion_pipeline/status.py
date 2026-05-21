"""Show current DB and queue status."""
import os
os.environ.setdefault("PATENT_DATA_DIR", "./data")
from src.patent_pipeline.database import PatentDatabase

db = PatentDatabase()
pending = db.connection.execute("SELECT COUNT(*) FROM parse_queue WHERE status='pending'").fetchone()[0]
running = db.connection.execute("SELECT COUNT(*) FROM parse_queue WHERE status='running'").fetchone()[0]
done = db.connection.execute("SELECT COUNT(*) FROM parse_queue WHERE status='done'").fetchone()[0]
errors = db.connection.execute("SELECT COUNT(*) FROM parse_queue WHERE status='error'").fetchone()[0]
total_rxns = db.connection.execute("SELECT COUNT(*) FROM reactions").fetchone()[0]
patent_rxns = db.connection.execute("SELECT COUNT(*) FROM reactions WHERE source='patent'").fetchone()[0]
ord_rxns = db.connection.execute("SELECT COUNT(*) FROM reactions WHERE source='ord'").fetchone()[0]
patents_count = db.connection.execute("SELECT COUNT(*) FROM patents").fetchone()[0]
print(f"Queue: pending={pending}, running={running}, done={done}, errors={errors}")
print(f"Reactions: {total_rxns} total ({patent_rxns} LLM-extracted, {ord_rxns} ORD)")
print(f"Patents: {patents_count}")
db.close()
