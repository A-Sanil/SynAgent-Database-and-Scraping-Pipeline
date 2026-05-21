"""Reset stuck queue items and show status."""
import os
os.environ.setdefault("PATENT_DATA_DIR", "./data")
from src.patent_pipeline.database import PatentDatabase
db = PatentDatabase()
n = db.connection.execute(
    "UPDATE parse_queue SET status='pending' WHERE status IN ('running','error')"
).rowcount
db.connection.commit()
pending = db.connection.execute(
    "SELECT COUNT(*) FROM parse_queue WHERE status='pending'"
).fetchone()[0]
done = db.connection.execute(
    "SELECT COUNT(*) FROM parse_queue WHERE status='done'"
).fetchone()[0]
print(f"Reset {n} stuck items. Pending: {pending}, Done: {done}")
db.close()
