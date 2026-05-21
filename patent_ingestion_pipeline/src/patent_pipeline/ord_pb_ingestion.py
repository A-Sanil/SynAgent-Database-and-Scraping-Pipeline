"""Ingest reactions from ORD .pb.gz files (Protocol Buffer format).

Downloads .pb.gz datasets from the ORD GitHub repository via Git LFS and
inserts reactions directly into the database — no LLM needed.

Usage:
    from patent_pipeline.ord_pb_ingestion import ingest_ord_pb_files
    from patent_pipeline.database import PatentDatabase

    db = PatentDatabase()
    n = ingest_ord_pb_files(db, pb_dir="ord_data", limit=5000)
    print(f"{n} reactions inserted")
    db.close()
"""

from __future__ import annotations

import gzip
import json
import os
import urllib.request
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .database import PatentDatabase
    from .models import PatentRecord, ReactionRecord

ORD_GITHUB_LFS = (
    "https://media.githubusercontent.com/media/"
    "open-reaction-database/ord-data/main/"
)
ORD_GITHUB_API = (
    "https://api.github.com/repos/open-reaction-database/ord-data/"
    "git/trees/main?recursive=1"
)
ORD_PARENT_PATENT_ID = "ORD-DATASET"

# CompoundIdentifier.Type enum values from ord-schema
_SMILES_TYPE = 2
_INCHI_TYPE  = 3
_NAME_TYPE   = 6


# ---------------------------------------------------------------------------
# ORD field extractors
# ---------------------------------------------------------------------------

def _get_smiles(compound) -> str | None:
    for ident in compound.identifiers:
        if ident.type == _SMILES_TYPE and ident.value:
            return ident.value
    return None


def _get_reactant_smiles(reaction) -> list[str]:
    smiles_list: list[str] = []
    for comp_group in reaction.inputs.values():
        for compound in comp_group.components:
            s = _get_smiles(compound)
            if s:
                smiles_list.append(s)
    return smiles_list


def _get_product_smiles(reaction) -> str | None:
    for outcome in reaction.outcomes:
        for product in outcome.products:
            s = _get_smiles(product)
            if s:
                return s
    return None


def _get_yield(reaction) -> float | None:
    from ord_schema.proto import reaction_pb2
    YIELD_TYPE = reaction_pb2.ProductMeasurement.YIELD
    for outcome in reaction.outcomes:
        for product in outcome.products:
            for meas in product.measurements:
                if meas.type == YIELD_TYPE:
                    if meas.HasField("percentage"):
                        return float(meas.percentage.value)
    return None


def _get_temperature(reaction) -> float | None:
    """Return temperature in Celsius."""
    from ord_schema.proto import reaction_pb2
    cond = reaction.conditions.temperature
    if not cond.HasField("setpoint"):
        return None
    val = float(cond.setpoint.value)
    unit = cond.setpoint.units
    CELSIUS   = reaction_pb2.Temperature.Celsius.CELSIUS
    FAHRENHEIT= reaction_pb2.Temperature.Celsius.FAHRENHEIT
    KELVIN    = reaction_pb2.Temperature.Celsius.KELVIN
    if unit == FAHRENHEIT:
        return (val - 32) * 5 / 9
    if unit == KELVIN:
        return val - 273.15
    return val  # Celsius


def _get_solvents(reaction) -> str | None:
    solvents = []
    for comp_group in reaction.inputs.values():
        for compound in comp_group.components:
            from ord_schema.proto import reaction_pb2
            if compound.reaction_role == reaction_pb2.ReactionRole.SOLVENT:
                s = _get_smiles(compound)
                if s:
                    solvents.append(s)
    if not solvents:
        for cond_solvent in reaction.conditions.solvents:
            s = _get_smiles(cond_solvent)
            if s:
                solvents.append(s)
    return ", ".join(solvents) if solvents else None


def _get_catalysts(reaction) -> str | None:
    cats = []
    for comp_group in reaction.inputs.values():
        for compound in comp_group.components:
            from ord_schema.proto import reaction_pb2
            if compound.reaction_role in (
                reaction_pb2.ReactionRole.CATALYST,
                reaction_pb2.ReactionRole.REAGENT,
            ):
                s = _get_smiles(compound)
                if s:
                    cats.append(s)
    return ", ".join(cats) if cats else None


def _reaction_to_record(rxn, dataset_name: str) -> "ReactionRecord | None":
    from .models import ReactionRecord

    rid = rxn.reaction_id or f"ord-{uuid.uuid4().hex[:16]}"
    reactants = _get_reactant_smiles(rxn)
    product   = _get_product_smiles(rxn)

    # Build reaction SMILES string
    reaction_smarts: str | None = None
    if reactants or product:
        r_part = ".".join(reactants) if reactants else ""
        p_part = product or ""
        reaction_smarts = f"{r_part}>>{p_part}"

    try:
        yld  = _get_yield(rxn)
        temp = _get_temperature(rxn)
        solv = _get_solvents(rxn)
        cat  = _get_catalysts(rxn)
    except Exception:
        yld = temp = solv = cat = None

    return ReactionRecord(
        reaction_id=rid,
        patent_id=ORD_PARENT_PATENT_ID,
        reaction_smarts=reaction_smarts,
        reactant_smiles=reactants,
        product_smiles=product,
        yield_percent=yld,
        temperature_celsius=temp,
        solvent=solv,
        catalyst=cat,
        notes=f"ORD dataset: {dataset_name}",
        metadata={"source": "ord", "confidence": 1.0, "dataset": dataset_name},
    )


# ---------------------------------------------------------------------------
# Parent patent record
# ---------------------------------------------------------------------------

def _ensure_ord_parent(db: "PatentDatabase") -> None:
    from .models import PatentRecord
    exists = db.connection.execute(
        "SELECT 1 FROM patents WHERE patent_id = ?", (ORD_PARENT_PATENT_ID,)
    ).fetchone()
    if exists:
        return
    record = PatentRecord(
        patent_id=ORD_PARENT_PATENT_ID,
        title="Open Reaction Database (ORD) — Bulk Import",
        abstract=(
            "Reactions imported directly from the Open Reaction Database. "
            "All SMILES, yields, temperatures, solvents, and catalysts are "
            "expert-curated and require no LLM re-parsing."
        ),
        source_url="https://open-reaction-database.org/",
        domain_tags=["ord", "curated", "bulk-import"],
        metadata={"source": "ord"},
    )
    db.upsert_patent(record)


# ---------------------------------------------------------------------------
# File download helpers
# ---------------------------------------------------------------------------

def list_ord_pb_files(limit: int = 20) -> list[str]:
    """Return up to `limit` .pb.gz relative paths from the ORD GitHub tree."""
    with urllib.request.urlopen(ORD_GITHUB_API, timeout=15) as r:
        data = json.loads(r.read())
    return [
        f["path"]
        for f in data.get("tree", [])
        if f["path"].endswith(".pb.gz")
    ][:limit]


def download_ord_pb_files(
    dest_dir: str | Path,
    n_files: int = 10,
    verbose: bool = True,
) -> list[Path]:
    """Download up to n_files ORD .pb.gz files via GitHub LFS."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    already = {p.name for p in dest_dir.glob("*.pb.gz")}
    remote_paths = list_ord_pb_files(limit=n_files + len(already) + 5)

    downloaded: list[Path] = list(dest_dir.glob("*.pb.gz"))
    for rel_path in remote_paths:
        if len(downloaded) >= n_files:
            break
        name = os.path.basename(rel_path)
        dest = dest_dir / name
        if name in already:
            continue
        url = ORD_GITHUB_LFS + rel_path
        if verbose:
            print(f"[ORD] Downloading {name} …", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            size_kb = dest.stat().st_size // 1024
            if verbose:
                print(f"{size_kb} KB")
            downloaded.append(dest)
        except Exception as exc:
            if verbose:
                print(f"FAILED: {exc}")
    return downloaded


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def ingest_ord_pb_files(
    db: "PatentDatabase",
    pb_dir: str | Path = "ord_data",
    n_files: int = 10,
    limit: int = 5000,
    download: bool = True,
    verbose: bool = True,
) -> int:
    """Parse ORD .pb.gz files and insert reactions into the database.

    Args:
        db: Open PatentDatabase instance.
        pb_dir: Directory containing (or to download) .pb.gz files.
        n_files: Maximum number of .pb.gz files to process.
        limit: Maximum total reactions to insert.
        download: Download missing files from GitHub LFS if True.
        verbose: Print progress.

    Returns:
        Number of reactions inserted.
    """
    try:
        from ord_schema.proto import dataset_pb2
    except ImportError:
        raise RuntimeError(
            "ord-schema is required: pip install ord-schema"
        )

    pb_dir = Path(pb_dir)
    _ensure_ord_parent(db)

    # Ensure source column exists
    cols = {row[1] for row in db.connection.execute("PRAGMA table_info(reactions)").fetchall()}
    if "source" not in cols:
        with db.connection:
            db.connection.execute("ALTER TABLE reactions ADD COLUMN source TEXT DEFAULT 'patent'")

    # Download files if needed
    if download:
        existing = list(pb_dir.glob("*.pb.gz"))
        if len(existing) < n_files:
            if verbose:
                print(f"[ORD] Downloading {n_files} .pb.gz files from ORD GitHub …")
            download_ord_pb_files(pb_dir, n_files=n_files, verbose=verbose)

    pb_files = sorted(pb_dir.glob("*.pb.gz"))[:n_files]
    if not pb_files:
        raise RuntimeError(f"No .pb.gz files found in {pb_dir}")

    if verbose:
        print(f"[ORD] Processing {len(pb_files)} files (limit={limit:,} reactions) …")

    inserted = 0
    errors   = 0

    for pb_path in pb_files:
        if inserted >= limit:
            break
        if verbose:
            print(f"[ORD]   {pb_path.name} …", end=" ", flush=True)
        try:
            with gzip.open(pb_path, "rb") as fh:
                raw = fh.read()
            ds = dataset_pb2.Dataset()
            ds.ParseFromString(raw)
        except Exception as exc:
            if verbose:
                print(f"SKIP (parse error: {exc})")
            continue

        file_inserted = 0
        for rxn in ds.reactions:
            if inserted >= limit:
                break
            try:
                record = _reaction_to_record(rxn, ds.name)
                if record is None:
                    continue
                with db.connection:
                    db._insert_reaction(record)
                    db.connection.execute(
                        "UPDATE reactions SET source = 'ord' WHERE reaction_id = ?",
                        (record.reaction_id,),
                    )
                inserted   += 1
                file_inserted += 1
            except Exception:
                errors += 1

        if verbose:
            print(f"{len(ds.reactions)} rxns -> inserted {file_inserted}")

    if verbose:
        print(f"[ORD] Complete: {inserted:,} reactions inserted ({errors} errors).")

    return inserted
