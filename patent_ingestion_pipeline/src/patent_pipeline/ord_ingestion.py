"""Ingest reactions from the Open Reaction Database (ORD) via HuggingFace Hub.

ORD data is already expert-curated: SMILES, yields, temperatures, and conditions
are extracted by chemists, so no LLM parsing is needed.  Records are inserted
directly into the reactions table with source='ord'.

The dataset is large (50+ GB full corpus); use the `limit` parameter to pull
a manageable subset (5 000–10 000 reactions recommended for initial testing).

Usage:
    from patent_pipeline.ord_ingestion import ingest_ord_dataset
    from patent_pipeline.database import PatentDatabase

    db = PatentDatabase()
    n = ingest_ord_dataset(db, limit=5000)
    print(f"Inserted {n} ORD reactions")
    db.close()
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .database import PatentDatabase
    from .models import ReactionRecord

ORD_HUGGINGFACE_ID = "open-reaction-database/ord-data"
ORD_PARENT_PATENT_ID = "ORD-DATASET"


# ---------------------------------------------------------------------------
# Parent record helpers
# ---------------------------------------------------------------------------

def _ensure_ord_parent_patent(db: "PatentDatabase") -> None:
    """Insert a synthetic patent record that owns all imported ORD reactions."""
    from .models import PatentRecord

    existing = db.connection.execute(
        "SELECT 1 FROM patents WHERE patent_id = ?", (ORD_PARENT_PATENT_ID,)
    ).fetchone()
    if existing:
        return
    record = PatentRecord(
        patent_id=ORD_PARENT_PATENT_ID,
        title="Open Reaction Database (ORD) — Bulk Import",
        abstract=(
            "Reactions imported directly from the Open Reaction Database. "
            "All SMILES, yields, temperatures, solvents, and catalysts were "
            "curated by domain experts and require no LLM re-parsing."
        ),
        source_url="https://open-reaction-database.org/",
        domain_tags=["ord", "curated", "bulk-import"],
        metadata={"source": "ord"},
    )
    db.upsert_patent(record)


# ---------------------------------------------------------------------------
# Field extraction helpers (schema-agnostic)
# ---------------------------------------------------------------------------

def _pick(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return default


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _list_to_str(v: Any) -> str | None:
    if isinstance(v, list):
        joined = ", ".join(str(x) for x in v if x)
        return joined or None
    if isinstance(v, str):
        return v or None
    return None


def _parse_reaction_smiles(rxn_smiles: str | None) -> tuple[list[str], str | None]:
    """Split 'reactants>>agents>>products' into (reactant_list, first_product)."""
    if not rxn_smiles:
        return [], None
    parts = rxn_smiles.split(">>")
    reactants = [s for s in parts[0].split(".") if s] if parts[0] else []
    products_str = parts[-1] if len(parts) >= 2 else ""
    product = products_str.split(".")[0] if products_str else None
    return reactants, product or None


def _example_to_reaction(example: dict[str, Any]) -> "ReactionRecord | None":
    """Convert one HuggingFace ORD row to a ReactionRecord.

    The function is tolerant of different column name conventions used by
    different ORD export formats on HuggingFace.
    """
    from .models import ReactionRecord

    reaction_id = str(
        _pick(example, "reaction_id", "id", "ord_id")
        or f"ord-{uuid.uuid4().hex[:16]}"
    )

    reaction_smarts: str | None = _pick(
        example, "reaction_smiles", "reaction_smarts", "mapped_rxn", "rxn_smiles"
    )

    # Reactants / product — try explicit fields first, then parse from SMILES.
    reactant_smiles: list[str] = []
    product_smiles: str | None = None

    raw_reactants = _pick(example, "reactant_smiles", "reactants_smiles", "reactants")
    raw_products = _pick(example, "product_smiles", "products_smiles", "products")

    if raw_reactants:
        if isinstance(raw_reactants, list):
            reactant_smiles = [str(s) for s in raw_reactants if s]
        elif isinstance(raw_reactants, str):
            reactant_smiles = [s.strip() for s in raw_reactants.split(".") if s.strip()]

    if raw_products:
        if isinstance(raw_products, list) and raw_products:
            product_smiles = str(raw_products[0])
        elif isinstance(raw_products, str):
            product_smiles = raw_products or None

    if not reactant_smiles or product_smiles is None:
        parsed_r, parsed_p = _parse_reaction_smiles(reaction_smarts)
        reactant_smiles = reactant_smiles or parsed_r
        product_smiles = product_smiles or parsed_p

    yield_pct = _to_float(_pick(example, "yield", "yield_percent", "yield_percentage", "product_yield"))
    temp_c = _to_float(_pick(example, "temperature_c", "temperature_celsius", "temperature", "temp_c"))
    time_h = _to_float(_pick(example, "time_hours", "reaction_time_h", "time_h", "duration_hours"))

    solvent = _list_to_str(_pick(example, "solvent", "solvents"))
    catalyst = _list_to_str(
        _pick(example, "catalyst", "catalysts", "reagent", "reagents", "agent", "agents")
    )

    return ReactionRecord(
        reaction_id=reaction_id,
        patent_id=ORD_PARENT_PATENT_ID,
        reaction_smarts=reaction_smarts,
        reactant_smiles=reactant_smiles,
        product_smiles=product_smiles,
        yield_percent=yield_pct,
        temperature_celsius=temp_c,
        solvent=solvent,
        catalyst=catalyst,
        time_hours=time_h,
        notes="Imported from Open Reaction Database (ORD)",
        metadata={"source": "ord", "confidence": 1.0},
    )


# ---------------------------------------------------------------------------
# Public ingestion entry point
# ---------------------------------------------------------------------------

def ingest_ord_dataset(
    db: "PatentDatabase",
    limit: int = 5000,
    dataset_id: str = ORD_HUGGINGFACE_ID,
    split: str = "train",
    verbose: bool = True,
) -> int:
    """Stream ORD reactions from HuggingFace and insert them into the database.

    Args:
        db: Open PatentDatabase instance.
        limit: Maximum reactions to insert (default 5 000; full corpus is ~1 M+).
        dataset_id: HuggingFace dataset identifier.
        split: Dataset split to stream (default 'train').
        verbose: Print progress every 500 reactions.

    Returns:
        Number of reactions successfully inserted.

    Raises:
        RuntimeError: If the 'datasets' package is missing or the dataset fails to load.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "HuggingFace 'datasets' package is required:\n"
            "    pip install datasets\n"
        )

    if verbose:
        print(f"[ORD] Streaming '{dataset_id}' split='{split}' (limit={limit:,}) …")

    _ensure_ord_parent_patent(db)

    try:
        ds = load_dataset(
            dataset_id,
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load ORD dataset from HuggingFace: {exc}\n"
            "Tip: check that the dataset ID is correct and you have internet access."
        ) from exc

    inserted = 0
    errors = 0

    with db.connection:
        for example in ds:
            if inserted >= limit:
                break
            try:
                reaction = _example_to_reaction(example)
                if reaction is None:
                    continue
                # Use the database's internal insert, tagging source='ord'
                db._insert_reaction(reaction)
                # Update source column right after insert (avoids changing _insert_reaction signature)
                db.connection.execute(
                    "UPDATE reactions SET source = 'ord' WHERE reaction_id = ?",
                    (reaction.reaction_id,),
                )
                inserted += 1
                if verbose and inserted % 500 == 0:
                    print(f"[ORD]   {inserted:,} / {limit:,} reactions inserted …")
            except Exception:
                errors += 1
                if errors > 200 and inserted == 0:
                    raise RuntimeError(
                        "More than 200 consecutive errors with zero successful inserts. "
                        "The dataset schema may differ from expected — inspect the first "
                        "example manually."
                    )

    if verbose:
        print(f"[ORD] Done. {inserted:,} reactions inserted ({errors} skipped with errors).")
    return inserted
