"""Helpers to query PubChem for name/SMILES enrichment and canonicalization.

Uses PubChem PUG-REST APIs. Optional network dependency; functions fail gracefully.
"""
from __future__ import annotations

import requests
import urllib.parse
from typing import Optional

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def name_to_canonical_smiles(name: str) -> Optional[str]:
    try:
        name_q = urllib.parse.quote(name)
        url = f"{PUBCHEM_BASE}/compound/name/{name_q}/property/CanonicalSMILES/JSON"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        props = data.get('PropertyTable', {}).get('Properties', [])
        if props:
            return props[0].get('CanonicalSMILES')
    except Exception:
        return None
    return None


def smiles_to_canonical_smiles(smiles: str) -> Optional[str]:
    try:
        s_q = urllib.parse.quote(smiles)
        # Ask for CID from smiles
        url = f"{PUBCHEM_BASE}/compound/smiles/{s_q}/cids/JSON"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        cids = data.get('IdentifierList', {}).get('CID', [])
        if not cids:
            return None
        cid = cids[0]
        url2 = f"{PUBCHEM_BASE}/compound/cid/{cid}/property/CanonicalSMILES/JSON"
        r2 = requests.get(url2, timeout=10)
        r2.raise_for_status()
        data2 = r2.json()
        props = data2.get('PropertyTable', {}).get('Properties', [])
        if props:
            return props[0].get('CanonicalSMILES')
    except Exception:
        return None
    return None
