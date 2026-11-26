#!/usr/bin/env python3
"""
pdb_selector.py
===============
Select and download the best PDB structure for a UniProt accession.

Ranks by: coverage × resolution (highest coverage, best resolution wins).
Prefers X-ray over EM. Downloads PDB file ready for RINmaker.

Usage:
    from pdb_selector import select_best_pdb, download_pdb

    best = select_best_pdb("P07949")
    pdb_path = download_pdb(best["pdb_id"], out_dir="./structures")
"""

import os
import json
import time
import requests


PDBE_BASE = "https://www.ebi.ac.uk/pdbe/api"
RCSB_DOWNLOAD = "https://files.rcsb.org/download"


def _get_json(url, timeout=30):
    """Fetch JSON from URL, return None on failure."""
    try:
        r = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def get_pdb_mappings(acc, cache_dir=None):
    """Get PDB entries mapped to a UniProt accession."""
    acc = acc.strip().upper()

    if cache_dir:
        cached = os.path.join(cache_dir, f"{acc}_pdb_mappings.json")
        if os.path.exists(cached):
            with open(cached) as f:
                return json.load(f)

    data = _get_json(f"{PDBE_BASE}/mappings/{acc}")
    if not data or acc not in data:
        result = {}
    else:
        result = data[acc].get("PDB", {})

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, f"{acc}_pdb_mappings.json"), "w") as f:
            json.dump(result, f)

    return result


def get_resolution(pdb_id, cache_dir=None):
    """Get resolution and experimental method for a PDB entry."""
    pdb_id = pdb_id.lower().strip()

    if cache_dir:
        cached = os.path.join(cache_dir, f"{pdb_id}_experiment.json")
        if os.path.exists(cached):
            with open(cached) as f:
                return json.load(f)

    data = _get_json(f"{PDBE_BASE}/pdb/entry/experiment/{pdb_id}")
    if not data or pdb_id not in data:
        result = {"resolution": None, "method": "unknown"}
    else:
        exp = data[pdb_id]
        if isinstance(exp, list) and exp:
            exp = exp[0]
        result = {
            "resolution": exp.get("resolution"),
            "method": exp.get("experimental_method_class", "unknown"),
        }

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, f"{pdb_id}_experiment.json"), "w") as f:
            json.dump(result, f)

    return result


def select_best_pdb(acc, cache_dir=None, prefer_xray=True, min_coverage=50,
                    max_resolution=4.0, max_entries=20):
    """
    Select the best PDB structure for a UniProt accession.

    Ranking criteria:
      1. Filter: coverage >= min_coverage, resolution <= max_resolution
      2. Prefer X-ray over EM (if prefer_xray=True)
      3. Sort by: coverage descending, then resolution ascending
      4. Return best entry with chain, coverage, resolution

    Parameters
    ----------
    acc : str
        UniProt accession
    cache_dir : str
        Cache directory for API responses
    prefer_xray : bool
        Prefer X-ray structures over EM
    min_coverage : int
        Minimum UniProt coverage in residues
    max_resolution : float
        Maximum resolution in Angstroms
    max_entries : int
        Maximum PDB entries to check resolution for

    Returns
    -------
    dict with keys: pdb_id, chain_id, resolution, method, unp_start, unp_end,
                    coverage, auth_start, download_url
    None if no suitable structure found
    """
    acc = acc.strip().upper()
    pdb_entries = get_pdb_mappings(acc, cache_dir=cache_dir)

    if not pdb_entries:
        return None

    # Collect all chain mappings
    candidates = []
    for pdb_id, chains in pdb_entries.items():
        for chain_info in chains:
            unp_start = chain_info.get("unp_start", 0)
            unp_end = chain_info.get("unp_end", 0)
            coverage = unp_end - unp_start + 1

            if coverage < min_coverage:
                continue

            chain_id = chain_info.get("chain_id", "A")
            start_info = chain_info.get("start", {})
            auth_start = start_info.get("author_residue_number")
            if auth_start is None:
                auth_start = unp_start

            candidates.append({
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "unp_start": unp_start,
                "unp_end": unp_end,
                "coverage": coverage,
                "auth_start": auth_start,
            })

    if not candidates:
        return None

    # Sort by coverage descending, take top N for resolution check
    candidates.sort(key=lambda x: -x["coverage"])
    candidates = candidates[:max_entries]

    # Get resolution for unique PDB IDs
    seen_pdbs = set()
    resolution_cache = {}
    for c in candidates:
        if c["pdb_id"] not in seen_pdbs:
            seen_pdbs.add(c["pdb_id"])
            exp = get_resolution(c["pdb_id"], cache_dir=cache_dir)
            resolution_cache[c["pdb_id"]] = exp
            time.sleep(0.1)

    # Enrich candidates with resolution and method
    filtered = []
    for c in candidates:
        exp = resolution_cache.get(c["pdb_id"], {})
        c["resolution"] = exp.get("resolution")
        c["method"] = exp.get("method", "unknown")

        # Filter by resolution (allow None for EM without reported resolution)
        if c["resolution"] is not None and c["resolution"] > max_resolution:
            continue

        filtered.append(c)

    if not filtered:
        # Relax: try without resolution filter
        filtered = candidates

    if not filtered:
        return None

    # Sort: prefer X-ray, then by coverage desc, then resolution asc
    def sort_key(c):
        method_score = 0 if (prefer_xray and c["method"] == "x-ray") else 1
        res = c["resolution"] if c["resolution"] is not None else 10.0
        return (method_score, -c["coverage"], res)

    filtered.sort(key=sort_key)
    best = filtered[0]

    best["download_url"] = f"{RCSB_DOWNLOAD}/{best['pdb_id']}.pdb"
    best["uniprot_acc"] = acc

    return best


def download_pdb(pdb_id, out_dir="./structures", format="pdb"):
    """
    Download a PDB file from RCSB.

    Parameters
    ----------
    pdb_id : str
        4-character PDB ID
    out_dir : str
        Output directory
    format : str
        "pdb" or "cif"

    Returns
    -------
    str: path to downloaded file
    """
    pdb_id = pdb_id.lower().strip()
    os.makedirs(out_dir, exist_ok=True)

    if format == "cif":
        url = f"{RCSB_DOWNLOAD}/{pdb_id}.cif"
        out_path = os.path.join(out_dir, f"{pdb_id}.cif")
    else:
        url = f"{RCSB_DOWNLOAD}/{pdb_id}.pdb"
        out_path = os.path.join(out_dir, f"{pdb_id}.pdb")

    if os.path.exists(out_path):
        return out_path

    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download {pdb_id}: HTTP {r.status_code}")

    with open(out_path, "wb") as f:
        f.write(r.content)

    return out_path


def select_and_download(acc, out_dir="./structures", cache_dir=None, **kwargs):
    """
    Select best PDB structure and download it.

    Returns (pdb_path, selection_info) or (None, None) if no structure found.
    """
    info = select_best_pdb(acc, cache_dir=cache_dir, **kwargs)
    if info is None:
        return None, None

    pdb_path = download_pdb(info["pdb_id"], out_dir=out_dir)
    return pdb_path, info


# ═══════════════════════════════════════════════
# CLI for testing
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    accs = sys.argv[1:] if len(sys.argv) > 1 else ["P07949", "P04637", "P00533", "Q16512"]

    for acc in accs:
        print(f"\n{'=' * 50}")
        print(f"  {acc}")
        print(f"{'=' * 50}")

        info = select_best_pdb(acc, cache_dir="./pdb_cache")
        if info is None:
            print(f"  No suitable PDB structure found")
            continue

        res_str = f"{info['resolution']:.1f}Å" if info['resolution'] else "N/A"
        print(f"  Best: {info['pdb_id']} chain {info['chain_id']}")
        print(f"  Method: {info['method']}")
        print(f"  Resolution: {res_str}")
        print(f"  UniProt: {info['unp_start']}-{info['unp_end']} ({info['coverage']} residues)")
        print(f"  Download: {info['download_url']}")
