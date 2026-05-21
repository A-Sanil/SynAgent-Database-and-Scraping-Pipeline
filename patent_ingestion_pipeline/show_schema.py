"""Show database location, schema, and row counts."""
import os, sys
os.environ.setdefault("PATENT_DATA_DIR", "./data")
sys.stderr = open(os.devnull, "w")

from src.patent_pipeline.database import PatentDatabase

db = PatentDatabase()
print("DB FILE:", db.db_path)
print()

tables = db.connection.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall()

for (name,) in tables:
    count = db.connection.execute(f"SELECT COUNT(*) FROM [{name}]").fetchone()[0]
    print(f"TABLE: {name}  ({count:,} rows)")
    cols = db.connection.execute(f"PRAGMA table_info([{name}])").fetchall()
    for c in cols:
        pk = " [PK]" if c[5] else ""
        nn = " NOT NULL" if c[3] else ""
        print(f"    {c[1]:<30} {c[2]}{pk}{nn}")
    print()

db.close()
