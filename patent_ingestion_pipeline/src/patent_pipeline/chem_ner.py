"""Chemistry extraction helpers: NER, name->SMILES, normalization and validation.

This module uses optional dependencies (`chemdataextractor`, `opsin`, `rdkit`) when
available and falls back to heuristics otherwise.
"""

from __future__ import annotations

import json
import re
from typing import Any

try:
    from chemdataextractor import Document as CDEDocument
except Exception:
    CDEDocument = None

try:
    import opsin
except Exception:
    opsin = None

try:
    from rdkit import Chem
except Exception:
    Chem = None
    
try:
    from .pubchem import name_to_canonical_smiles, smiles_to_canonical_smiles
except Exception:
    # relative import might fail in some contexts; try direct import
    try:
        from patent_pipeline.pubchem import name_to_canonical_smiles, smiles_to_canonical_smiles
    except Exception:
        name_to_canonical_smiles = None
        smiles_to_canonical_smiles = None


YIELD_RE = re.compile(r"yield[s]?:?\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%")
TEMP_RE = re.compile(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*°?C")
SMILES_RE = re.compile(r"\b([A-Za-z0-9@+\-\[\]\(\)=#$]{5,})\b")


def name_to_smiles(name: str) -> str | None:
    if opsin is not None:
        try:
            smi = opsin.convert(name)
            if smi:
                return smi
        except Exception:
            pass
    # fallback to PubChem name lookup if available
    if name_to_canonical_smiles is not None:
        try:
            smi = name_to_canonical_smiles(name)
            if smi:
                return smi
        except Exception:
            pass
    return None


def verify_smiles(smiles: str) -> dict:
    """Verify and canonicalize a SMILES string.

    Returns a dict: { 'canonical': str|None, 'pubchem_match': bool, 'rdkit_valid': bool }
    """
    out = {"canonical": None, "pubchem_match": False, "rdkit_valid": False}
    if not smiles:
        return out
    # try RDKit canonicalization
    if Chem is not None:
        try:
            m = Chem.MolFromSmiles(smiles)
            if m is not None:
                out["rdkit_valid"] = True
                out["canonical"] = Chem.MolToSmiles(m, isomericSmiles=True)
        except Exception:
            pass

    # try PubChem canonicalization and check agreement
    if smiles_to_canonical_smiles is not None:
        try:
            pc = smiles_to_canonical_smiles(smiles)
            if pc:
                out["pubchem_match"] = True
                # prefer RDKit canonical if present and equal, else use PubChem
                if out.get("canonical") is None:
                    out["canonical"] = pc
                else:
                    # both present: check equality
                    if out["canonical"].lower() == pc.lower():
                        out["pubchem_match"] = True
                    else:
                        # different canonical forms; keep RDKit but note mismatch
                        out["pubchem_match"] = False
        except Exception:
            pass

    return out


def normalize_smiles(smiles: str) -> str | None:
    if Chem is None:
        return smiles
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        return Chem.MolToSmiles(m, isomericSmiles=True)
    except Exception:
        return None


def extract_chem_entities(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"yields": [], "temperatures": [], "smiles": [], "names": []}
    if not text:
        return out

    # Use chemdataextractor if available for richer extraction
    if CDEDocument is not None:
        try:
            doc = CDEDocument(text)
            # chemdataextractor provides .chemical_entities and .records; keep simple
            chems = []
            for e in getattr(doc, "chemical_entities", []):
                chems.append(str(e))
            out["names"].extend(chems)
        except Exception:
            pass

    # heuristic yields
    for m in YIELD_RE.finditer(text):
        try:
            out["yields"].append(float(m.group(1)))
        except Exception:
            pass

    for m in TEMP_RE.finditer(text):
        try:
            out["temperatures"].append(float(m.group(1)))
        except Exception:
            pass

    # SMILES-like heuristic (this will catch many false positives; validate later)
    for m in SMILES_RE.finditer(text):
        token = m.group(1)
        if len(token) >= 5:
            out["smiles"].append(token)

    # Try to canonicalize found SMILES
    normalized = []
    for s in out["smiles"]:
        ns = normalize_smiles(s)
        if ns:
            normalized.append(ns)
    out["smiles_normalized"] = normalized

    # try name->smiles for names found
    name_smiles = {}
    for name in out["names"][:20]:
        try:
            smi = name_to_smiles(name)
            if smi:
                name_smiles[name] = smi
        except Exception:
            pass
    out["name_to_smiles"] = name_smiles

    return out
