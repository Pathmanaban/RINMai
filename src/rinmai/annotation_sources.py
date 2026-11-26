"""
annotation_sources.py
======================
Multi-source functional annotation fetcher + stratified protein sampler.

Sources:
  1. UniProt        — active sites, binding, PTMs, disulfide, glycosylation
  2. PDBe-KB        — interface residues, ligand sites, catalytic sites (from crystal structures)
  3. iPTMnet        — experimentally validated PTMs (phospho, ubiq, acetyl, SUMO, etc.)
  4. ELM            — functional linear motifs in disordered regions

Plus:
  - Stratified protein sampler (family-diverse, annotation-dense, multi-species)
  - Combined annotation builder that deduplicates across sources

Requires: requests, pandas, numpy, tqdm
"""

import os
import time
import json
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm


# ═══════════════════════════════════════════════
# Shared types
# ═══════════════════════════════════════════════

@dataclass
class FunctionalAnnotation:
    """Unified annotation record across all sources."""
    uniprot_acc: str
    position: int           # 1-based
    aa: str                 # single letter (or "?" if unknown)
    label: str              # ACT_SITE, BINDING, INTERFACE, MOD_RES, PTM_*, MOTIF, etc.
    source: str             # "uniprot", "pdbe_kb", "iptmnet", "PTMexchange", "elm"
    detail: str = ""        # free text (e.g. "Phosphoserine", "ATP binding", "LIG_FHA_1")
    evidence: str = ""      # evidence code or PDB id


def _get_json(url, retries=3, delay=0.5, timeout=30):
    """GET JSON from a URL with retries and rate-limit awareness."""
    import requests
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout,
                             headers={"User-Agent": "RIN-annotation-v3/1.0",
                                      "Accept": "application/json"})
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 2))
                time.sleep(wait)
                continue
            # Other errors
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                print(f"    [ERR] {url}: {e}")
    return None


# ═══════════════════════════════════════════════
# 1. UniProt fetcher (carried over from v2)
# ═══════════════════════════════════════════════

UNIPROT_FEATURE_MAP = {
    "Active site": "ACT_SITE",
    "Binding site": "BINDING",
    "Site": "SITE",
    "Modified residue": "MOD_RES",
    "Glycosylation": "GLYCO",
    "Lipidation": "LIPID",
    "Cross-link": "CROSSLINK",
    "Disulfide bond": "DISULFIDE",
}

UNIPROT_CACHE_VERSION = "v2"
UNIPROT_ENDPOINT_FEATURES = {"DISULFIDE", "CROSSLINK"}
UNIPROT_POINT_FEATURES = {
    "ACT_SITE", "BINDING", "SITE", "MOD_RES", "GLYCO", "LIPID",
}


def _as_int(value):
    """Return an int for API coordinate values, or None if unavailable."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _uniprot_feature_positions(label, start, end):
    """
    Convert UniProt feature coordinates into residue-level annotation positions.

    Site-like UniProt features are only kept when they resolve to one residue.
    Bond/cross-link features encode two endpoints as start/end, not a full span.
    """
    start = _as_int(start)
    end = _as_int(end)
    if start is None:
        return []
    if end is None:
        end = start

    if label in UNIPROT_ENDPOINT_FEATURES:
        return [start] if start == end else [start, end]

    if label in UNIPROT_POINT_FEATURES:
        return [start] if start == end else []

    return list(range(start, end + 1))


def fetch_uniprot_annotations(acc, cache_dir=None):
    """Fetch functional annotations from UniProt REST API."""
    acc = acc.strip().upper()

    # Check cache
    if cache_dir:
        cached = os.path.join(cache_dir, f"{acc}_uniprot_{UNIPROT_CACHE_VERSION}.json")
        if os.path.exists(cached):
            with open(cached) as f:
                return json.load(f)

    url = f"https://rest.uniprot.org/uniprotkb/{acc}?format=json"
    entry = _get_json(url)
    if entry is None:
        return []

    seq = entry.get("sequence", {}).get("value", "")
    results = []

    for feat in entry.get("features", []):
        label = UNIPROT_FEATURE_MAP.get(feat.get("type", ""))
        if label is None:
            continue
        loc = feat.get("location", {})
        start = loc.get("start", {}).get("value")
        end = loc.get("end", {}).get("value")
        if start is None:
            continue
        if end is None:
            end = start

        desc = feat.get("description", "") or ""
        ecos = [e.get("evidenceCode", "") for e in feat.get("evidences", [])]
        positions = _uniprot_feature_positions(label, start, end)

        for pos in positions:
            if pos < 1 or pos > len(seq):
                continue
            detail = desc
            if label in UNIPROT_ENDPOINT_FEATURES and len(positions) == 2:
                partner = positions[1] if pos == positions[0] else positions[0]
                detail = f"{desc}; paired_with={partner}" if desc else f"paired_with={partner}"
            results.append(asdict(FunctionalAnnotation(
                uniprot_acc=acc, position=pos, aa=seq[pos - 1],
                label=label, source="uniprot",
                detail=detail, evidence=";".join(ecos),
            )))

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cached, "w") as f:
            json.dump(results, f)

    return results


# ═══════════════════════════════════════════════
# 2. PDBe-KB fetcher
# ═══════════════════════════════════════════════

PDBE_KB_BASE = "https://www.ebi.ac.uk/pdbe/graph-api/uniprot"


def fetch_pdbe_kb_annotations(acc, cache_dir=None):
    """
    Fetch from PDBe-KB:
      - interface_residues  → INTERFACE label
      - ligand_sites        → LIGAND_BINDING label

    Response: {acc: {sequence, length, dataType, data: [{name, accession, residues: [{startIndex, startCode, ...}]}]}}
    """
    acc = acc.strip().upper()

    if cache_dir:
        cached = os.path.join(cache_dir, f"{acc}_pdbe_kb.json")
        if os.path.exists(cached):
            with open(cached) as f:
                return json.load(f)

    AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
           "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
           "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

    results = []

    for endpoint, label in [("interface_residues", "INTERFACE"), ("ligand_sites", "LIGAND_BINDING")]:
        raw = _get_json(f"{PDBE_KB_BASE}/{endpoint}/{acc}")
        if not raw or not isinstance(raw, dict):
            continue
        acc_data = raw.get(acc, {})
        if not isinstance(acc_data, dict):
            continue
        data_list = acc_data.get("data", [])
        if not isinstance(data_list, list):
            continue

        for partner in data_list:
            if not isinstance(partner, dict):
                continue
            partner_name = partner.get("name", "") or ""
            ligand_acc = partner.get("accession", "") or ""
            detail = partner_name or ligand_acc

            for res in partner.get("residues", []):
                if not isinstance(res, dict):
                    continue
                pos = res.get("startIndex")
                if pos is None:
                    continue
                aa3 = res.get("startCode", "")
                aa1 = AA3.get(aa3, "?")
                pdb_entries = res.get("interactingPDBEntries", [])
                pdb_ids = [e.get("pdbId", "") for e in pdb_entries if isinstance(e, dict)]

                results.append(asdict(FunctionalAnnotation(
                    uniprot_acc=acc, position=int(pos), aa=aa1,
                    label=label, source="pdbe_kb",
                    detail=detail,
                    evidence=",".join(pdb_ids[:3]),
                )))

        time.sleep(0.2)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cached, "w") as f:
            json.dump(results, f)

    return results


# ═══════════════════════════════════════════════
# 3. iPTMnet fetcher
# ═══════════════════════════════════════════════

IPTMNET_BASE = "https://research.bioinformatics.udel.edu/iptmnet/api"

# iPTMnet PTM type mapping
IPTMNET_TYPE_MAP = {
    "Phosphorylation": "PTM_PHOSPHO",
    "Ubiquitination": "PTM_UBIQ",
    "Acetylation": "PTM_ACETYL",
    "SUMOylation": "PTM_SUMO",
    "Sumoylation": "PTM_SUMO",
    "Methylation": "PTM_METHYL",
    "Glycosylation": "PTM_GLYCO",
    "O-Glycosylation": "PTM_GLYCO",
    "N-Glycosylation": "PTM_GLYCO",
    "S-Nitrosylation": "PTM_SNITRO",
    "Palmitoylation": "PTM_PALMI",
    "Myristoylation": "PTM_MYRIS",
}


def _ptm_label_from_name(ptm_name):
    """Normalize a PTM name to the labels used by the training pipeline."""
    ptm_name = (ptm_name or "").strip()
    if not ptm_name:
        return "PTM_OTHER"
    return IPTMNET_TYPE_MAP.get(ptm_name, f"PTM_{ptm_name.upper().replace(' ', '_')}")


def fetch_iptmnet_annotations(acc, cache_dir=None):
    """
    Fetch experimentally validated PTM sites from iPTMnet.

    Response: {acc: [records], acc-1: [records], ...}
    Each record: {residue: "C", site: "C317", ptm_type: "Methylation", score, sources, ...}
    site is a STRING like "C317" — amino acid letter prefixed to position number.
    """
    acc = acc.strip().upper()

    if cache_dir:
        cached = os.path.join(cache_dir, f"{acc}_iptmnet.json")
        if os.path.exists(cached):
            with open(cached) as f:
                return json.load(f)

    results = []

    data = _get_json(f"{IPTMNET_BASE}/{acc}/substrate", timeout=15)

    if data and isinstance(data, dict):
        # Get records for canonical accession
        records = data.get(acc, [])
        if not isinstance(records, list):
            records = []

        for record in records:
            if not isinstance(record, dict):
                continue

            ptm_type = record.get("ptm_type", "") or ""
            site_str = str(record.get("site", ""))

            # Parse site string: "C317" → aa="C", pos=317
            aa = site_str[0] if site_str and site_str[0].isalpha() else "?"
            pos_str = site_str.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

            try:
                pos = int(pos_str)
            except (ValueError, TypeError):
                continue

            label = _ptm_label_from_name(ptm_type)

            # Evidence sources
            sources = record.get("sources", [])
            if isinstance(sources, list):
                evidence_str = ";".join(
                    s.get("name", "") if isinstance(s, dict) else str(s)
                    for s in sources
                )
            else:
                evidence_str = ""

            results.append(asdict(FunctionalAnnotation(
                uniprot_acc=acc, position=pos, aa=aa,
                label=label, source="iptmnet",
                detail=ptm_type, evidence=evidence_str,
            )))

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cached, "w") as f:
            json.dump(results, f)

    return results


# ═══════════════════════════════════════════════
# 4. PTMExchange / EBI Proteomics PTM fetcher
# ═══════════════════════════════════════════════

PTMEXCHANGE_BASE = "https://www.ebi.ac.uk/proteins/api/proteomics/ptm"


def _extract_sequence_value(data):
    seq = data.get("sequence", "") if isinstance(data, dict) else ""
    if isinstance(seq, dict):
        return seq.get("value", "") or ""
    return seq or ""


def _ptmexchange_position(feature, ptm):
    """
    Convert PTMExchange peptide-relative PTM coordinates to UniProt positions.

    The feature begin/end define the peptide span on the protein. The nested
    ptms[].position value is the position within that peptide.
    """
    raw_pos = _as_int(ptm.get("position"))
    begin = _as_int(feature.get("begin"))
    end = _as_int(feature.get("end"))
    peptide = feature.get("peptide", "") or ""

    if raw_pos is None:
        return None, None
    if begin is not None and (not peptide or 1 <= raw_pos <= len(peptide)):
        pos = begin + raw_pos - 1
        if end is None or pos <= end:
            return pos, raw_pos
    return raw_pos, raw_pos


def _ptmexchange_evidence(feature, ptm):
    parts = []

    for source in ptm.get("sources", []) or []:
        parts.append(str(source))

    for ref in ptm.get("dbReferences", []) or []:
        if not isinstance(ref, dict):
            continue
        name = ref.get("name", "") or ""
        ref_id = ref.get("id", "") or ""
        if name or ref_id:
            parts.append(":".join(p for p in [name, ref_id] if p))

    for evidence in feature.get("evidences", []) or []:
        if not isinstance(evidence, dict):
            continue
        source = evidence.get("source", {}) or {}
        if not isinstance(source, dict):
            continue
        name = source.get("name", "") or ""
        source_id = source.get("id", "") or ""
        if name or source_id:
            parts.append(":".join(p for p in [name, source_id] if p))

    return ";".join(dict.fromkeys(p for p in parts if p))


def fetch_ptmexchange_annotations(acc, cache_dir=None):
    """
    Fetch MS-derived PTM sites from EBI Proteins proteomics/ptm.

    Response features are peptide spans; nested ptms[] entries provide the
    modification name and peptide-relative position.
    """
    acc = acc.strip().upper()

    if cache_dir:
        cached = os.path.join(cache_dir, f"{acc}_ptmexchange.json")
        if os.path.exists(cached):
            with open(cached) as f:
                return json.load(f)

    results = []
    data = _get_json(f"{PTMEXCHANGE_BASE}/{acc}", timeout=20)
    if data and isinstance(data, dict):
        seq = _extract_sequence_value(data)

        for feature in data.get("features", []) or []:
            if not isinstance(feature, dict):
                continue
            peptide = feature.get("peptide", "") or ""

            for ptm in feature.get("ptms", []) or []:
                if not isinstance(ptm, dict):
                    continue

                ptm_name = ptm.get("name", "") or ""
                pos, peptide_pos = _ptmexchange_position(feature, ptm)
                if pos is None or pos < 1:
                    continue

                aa = "?"
                if peptide and peptide_pos is not None and 1 <= peptide_pos <= len(peptide):
                    aa = peptide[peptide_pos - 1]
                elif seq and pos <= len(seq):
                    aa = seq[pos - 1]

                detail = ptm_name
                if peptide_pos is not None:
                    detail = f"{ptm_name}; peptide_pos={peptide_pos}" if ptm_name else f"peptide_pos={peptide_pos}"

                results.append(asdict(FunctionalAnnotation(
                    uniprot_acc=acc, position=pos, aa=aa,
                    label=_ptm_label_from_name(ptm_name),
                    source="PTMexchange",
                    detail=detail,
                    evidence=_ptmexchange_evidence(feature, ptm),
                )))

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cached, "w") as f:
            json.dump(results, f)

    return results


# 5. ELM fetcher

ELM_BASE = "http://elm.eu.org"


# ═══════════════════════════════════════════════
# PDB PTM annotations (from modified_AA_or_NA API)
# ═══════════════════════════════════════════════

# Map CCD codes to PTM type labels
PDB_PTM_TYPE_MAP = {
    # Phosphorylation
    "PTR": "PTM_PHOSPHO", "SEP": "PTM_PHOSPHO", "TPO": "PTM_PHOSPHO",
    "PHD": "PTM_PHOSPHO", "4TP": "PTM_PHOSPHO",
    # Acetylation
    "ALY": "PTM_ACETYL", "AYA": "PTM_ACETYL", "M2L": "PTM_ACETYL",
    # Methylation
    "MLY": "PTM_METHYL", "MLZ": "PTM_METHYL", "M3L": "PTM_METHYL",
    "2MR": "PTM_METHYL", "AGM": "PTM_METHYL",
    # Glycosylation
    "MAN": "PTM_GLYCO", "NAG": "PTM_GLYCO", "BMA": "PTM_GLYCO",
    "FUC": "PTM_GLYCO", "GAL": "PTM_GLYCO", "SIA": "PTM_GLYCO",
    # Lipidation
    "MYR": "PTM_MYRIS", "PLM": "PTM_PALMI",
    # Sulfation
    "TYS": "PTM_OTHER",
    # Hydroxylation
    "HYP": "PTM_OTHER", "HY3": "PTM_OTHER", "LYZ": "PTM_OTHER",
    # Pyroglutamic acid
    "PCA": "PTM_OTHER",
    # Citrullination
    "CIR": "PTM_OTHER",
    # Nitrosylation
    "SNC": "PTM_SNITRO",
    # Carboxylation
    "KCX": "PTM_OTHER", "CGU": "PTM_OTHER",
    # SUMOylation (rare in PDB)
    "UNK": None,  # skip unknown
}

# Map CCD codes to human-readable PTM names
PDB_PTM_NAME_MAP = {
    "PTR": "Phosphotyrosine", "SEP": "Phosphoserine", "TPO": "Phosphothreonine",
    "ALY": "N6-acetyllysine", "MLY": "N6,N6-dimethyllysine",
    "M3L": "N6,N6,N6-trimethyllysine", "HYP": "Hydroxyproline",
    "PCA": "Pyroglutamic acid", "TYS": "O-sulfotyrosine",
    "MYR": "Myristoylation", "PLM": "Palmitoylation",
    "SNC": "S-nitrosocysteine", "CIR": "Citrulline",
    "KCX": "Carboxylated lysine", "CGU": "Carboxyglutamic acid",
}

# Map CCD codes to parent amino acid
PDB_PTM_PARENT_AA = {
    "PTR": "Y", "SEP": "S", "TPO": "T", "PHD": "D",
    "ALY": "K", "AYA": "A", "MLY": "K", "MLZ": "K", "M3L": "K",
    "HYP": "P", "PCA": "E", "TYS": "Y", "SNC": "C", "CIR": "R",
    "KCX": "K", "CGU": "E", "MYR": "G", "PLM": "C",
    "2MR": "R", "AGM": "R", "M2L": "K",
}


def fetch_pdb_ptm_annotations(acc, cache_dir=None, max_pdb_entries=10):
    """
    Fetch experimentally observed PTMs from PDB structures via PDBe API.

    Uses two endpoints:
      1. /api/mappings/{acc} → get PDB entries + UniProt position mapping
      2. /api/pdb/entry/modified_AA_or_NA/{pdb_id} → get modified residues

    Maps PDB author_residue_number to UniProt position using the mapping data.
    """
    import requests
    acc = acc.strip().upper()

    if cache_dir:
        cached = os.path.join(cache_dir, f"{acc}_pdb_ptm.json")
        if os.path.exists(cached):
            with open(cached) as f:
                return json.load(f)

    results = []

    # Step 1: Get PDB mappings for this UniProt accession
    mappings_url = f"https://www.ebi.ac.uk/pdbe/api/mappings/{acc}"
    mappings_data = _get_json(mappings_url, timeout=30)
    if not mappings_data or acc not in mappings_data:
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, f"{acc}_pdb_ptm.json"), "w") as f:
                json.dump([], f)
        return []

    pdb_entries = mappings_data[acc].get("PDB", {})
    if not pdb_entries:
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, f"{acc}_pdb_ptm.json"), "w") as f:
                json.dump([], f)
        return []

    # Sort by coverage (largest unp range first), take top N
    pdb_ranked = []
    for pdb_id, chains in pdb_entries.items():
        for chain_info in chains:
            coverage = chain_info.get("unp_end", 0) - chain_info.get("unp_start", 0) + 1
            pdb_ranked.append((pdb_id, chain_info, coverage))
    pdb_ranked.sort(key=lambda x: -x[2])

    seen_pdb_ids = set()
    pdb_to_check = []
    for pdb_id, chain_info, cov in pdb_ranked:
        if pdb_id not in seen_pdb_ids and len(pdb_to_check) < max_pdb_entries:
            seen_pdb_ids.add(pdb_id)
            pdb_to_check.append((pdb_id, chain_info))

    # Step 2: For each PDB entry, get modified residues
    for pdb_id, chain_info in pdb_to_check:
        chain_id = chain_info.get("chain_id", "A")
        unp_start = chain_info.get("unp_start", 0)
        unp_end = chain_info.get("unp_end", 0)
        # Offset: author_residue_number → UniProt position
        start_info = chain_info.get("start", {})
        auth_start = start_info.get("author_residue_number")
        if auth_start is None:
            auth_start = unp_start  # assume author numbering = UniProt numbering
        offset = unp_start - int(auth_start)

        mod_url = f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/modified_AA_or_NA/{pdb_id}"
        mod_data = _get_json(mod_url, timeout=15)

        if not mod_data or pdb_id not in mod_data:
            time.sleep(0.2)
            continue

        for mod in mod_data[pdb_id]:
            if not isinstance(mod, dict):
                continue

            mod_chain = mod.get("chain_id", "")
            if mod_chain != chain_id:
                continue

            auth_resnum = mod.get("author_residue_number")
            ccd_id = mod.get("chem_comp_id", "")
            ccd_name = mod.get("chem_comp_name", "")

            if auth_resnum is None:
                continue

            # Map to UniProt position
            unp_pos = int(auth_resnum) + offset

            # Check if within UniProt range
            if unp_pos < unp_start or unp_pos > unp_end:
                continue

            # Map CCD to PTM label
            ptm_label = PDB_PTM_TYPE_MAP.get(ccd_id)
            if ptm_label is None:
                # Unknown CCD — try to classify from name
                name_upper = ccd_name.upper()
                if "PHOSPHO" in name_upper:
                    ptm_label = "PTM_PHOSPHO"
                elif "ACETYL" in name_upper:
                    ptm_label = "PTM_ACETYL"
                elif "METHYL" in name_upper:
                    ptm_label = "PTM_METHYL"
                elif "GLYCO" in name_upper or "SUGAR" in name_upper:
                    ptm_label = "PTM_GLYCO"
                else:
                    ptm_label = "PTM_OTHER"

            aa = PDB_PTM_PARENT_AA.get(ccd_id, "?")
            detail = PDB_PTM_NAME_MAP.get(ccd_id, ccd_name)

            results.append(asdict(FunctionalAnnotation(
                uniprot_acc=acc,
                position=unp_pos,
                aa=aa,
                label=ptm_label,
                source="pdb_ptm",
                detail=f"{detail} ({ccd_id})",
                evidence=f"PDB:{pdb_id}:{chain_id}",
            )))

        time.sleep(0.2)

    # Deduplicate (same position + label from different PDB entries)
    seen = set()
    unique_results = []
    for r in results:
        key = (r["uniprot_acc"], r["position"], r["label"])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, f"{acc}_pdb_ptm.json"), "w") as f:
            json.dump(unique_results, f)

    return unique_results


def fetch_elm_annotations(acc, cache_dir=None, annotated_only=True):
    """
    Fetch linear motif instances from ELM via TSV endpoint.

    URL: http://elm.eu.org/start_search/{acc}.tsv
    Columns: elm_identifier, start, stop, is_annotated, is_phiblastmatch,
             is_filtered, phiblast, topodomfilter, taxonfilter, structure

    By default only keeps is_annotated=True (experimentally validated).
    Set annotated_only=False to also include unfiltered predictions.
    """
    import io
    acc = acc.strip().upper()

    if cache_dir:
        cached = os.path.join(cache_dir, f"{acc}_elm.json")
        if os.path.exists(cached):
            with open(cached) as f:
                return json.load(f)

    results = []

    try:
        import requests
        url = f"{ELM_BASE}/start_search/{acc}.tsv"
        r = requests.get(url, timeout=15, headers={"User-Agent": "RIN-v3"})
        if r.status_code != 200:
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                with open(os.path.join(cache_dir, f"{acc}_elm.json"), "w") as f:
                    json.dump([], f)
            return []

        df = pd.read_csv(io.StringIO(r.text), sep="\t")

        if annotated_only:
            df = df[df["is_annotated"] == True]
        else:
            # Keep annotated + unfiltered predictions
            df = df[(df["is_annotated"] == True) | (df["is_filtered"] == False)]

        for _, row in df.iterrows():
            elm_id = str(row.get("elm_identifier", ""))
            start = row.get("start")
            stop = row.get("stop")
            is_validated = bool(row.get("is_annotated", False))

            if pd.isna(start):
                continue
            if pd.isna(stop):
                stop = start

            # Classify by ELM identifier prefix
            elm_upper = elm_id.upper()
            if elm_upper.startswith("LIG"):
                label = "MOTIF_LIG"
            elif elm_upper.startswith("MOD"):
                label = "MOTIF_MOD"
            elif elm_upper.startswith("DEG"):
                label = "MOTIF_DEG"
            elif elm_upper.startswith("DOC"):
                label = "MOTIF_DOC"
            elif elm_upper.startswith("CLV"):
                label = "MOTIF_CLV"
            elif elm_upper.startswith("TRG"):
                label = "MOTIF_TRG"
            else:
                label = "MOTIF_OTHER"

            evidence = "ELM:validated" if is_validated else "ELM:predicted"

            for pos in range(int(start), int(stop) + 1):
                results.append(asdict(FunctionalAnnotation(
                    uniprot_acc=acc, position=pos, aa="?",
                    label=label, source="elm",
                    detail=elm_id, evidence=evidence,
                )))

    except Exception as e:
        pass  # fail silently — ELM is a bonus source

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, f"{acc}_elm.json"), "w") as f:
            json.dump(results, f)

    return results


# ═══════════════════════════════════════════════
# 6. Combined annotation builder
# ═══════════════════════════════════════════════

# Which labels count as "functional" for training
# (updated with new sources)
TRAINING_LABELS = {
    # UniProt
    "ACT_SITE", "BINDING", "SITE", "MOD_RES", "GLYCO", "LIPID",
    # PDBe-KB
    "INTERFACE", "LIGAND_BINDING", "CATALYTIC",
    # iPTMnet / PTMExchange
    "PTM_PHOSPHO", "PTM_UBIQ", "PTM_ACETYL", "PTM_SUMO", "PTM_METHYL",
    "PTM_GLYCO", "PTM_SNITRO", "PTM_PALMI", "PTM_MYRIS", "PTM_OTHER",
    # ELM
    "MOTIF_LIG", "MOTIF_MOD", "MOTIF_DEG", "MOTIF_DOC",
    "MOTIF_CLV", "MOTIF_TRG", "MOTIF_OTHER",
}

# Excluded from training (structurally trivial or redundant)
EXCLUDED_LABELS = {"DISULFIDE", "CROSSLINK"}

# Broad categories for stratification / analysis
LABEL_CATEGORY = {
    "ACT_SITE": "catalytic",
    "CATALYTIC": "catalytic",
    "BINDING": "binding",
    "LIGAND_BINDING": "binding",
    "INTERFACE": "interface",
    "SITE": "functional_site",
    "MOD_RES": "ptm",
    "GLYCO": "ptm",
    "LIPID": "ptm",
    "PTM_PHOSPHO": "ptm",
    "PTM_UBIQ": "ptm",
    "PTM_ACETYL": "ptm",
    "PTM_SUMO": "ptm",
    "PTM_METHYL": "ptm",
    "PTM_GLYCO": "ptm",
    "PTM_SNITRO": "ptm",
    "PTM_PALMI": "ptm",
    "PTM_MYRIS": "ptm",
    "PTM_OTHER": "ptm",
    "MOTIF_LIG": "motif",
    "MOTIF_MOD": "motif",
    "MOTIF_DEG": "motif",
    "MOTIF_DOC": "motif",
    "MOTIF_CLV": "motif",
    "MOTIF_TRG": "motif",
    "MOTIF_OTHER": "motif",
    "DISULFIDE": "structural",
    "CROSSLINK": "structural",
}


def fetch_all_annotations(
    acc,
    sources=("uniprot", "pdbe_kb", "iptmnet", "ptmexchange", "pdb_ptm", "elm"),
    cache_dir="./annotation_cache",
    delay=0.3,
):
    """
    Fetch annotations from all specified sources for one protein.

    Returns list of annotation dicts.
    """
    all_annots = []

    if "uniprot" in sources:
        try:
            annots = fetch_uniprot_annotations(acc, cache_dir=cache_dir)
            all_annots.extend(annots)
        except Exception as e:
            print(f"    [WARN] UniProt failed for {acc}: {e}")

    if "pdbe_kb" in sources:
        try:
            annots = fetch_pdbe_kb_annotations(acc, cache_dir=cache_dir)
            all_annots.extend(annots)
            time.sleep(delay)
        except Exception as e:
            print(f"    [WARN] PDBe-KB failed for {acc}: {e}")

    if "iptmnet" in sources:
        try:
            annots = fetch_iptmnet_annotations(acc, cache_dir=cache_dir)
            all_annots.extend(annots)
            time.sleep(delay)
        except Exception as e:
            print(f"    [WARN] iPTMnet failed for {acc}: {e}")

    if "ptmexchange" in sources or "PTMexchange" in sources:
        try:
            annots = fetch_ptmexchange_annotations(acc, cache_dir=cache_dir)
            all_annots.extend(annots)
            time.sleep(delay)
        except Exception as e:
            print(f"    [WARN] PTMExchange failed for {acc}: {e}")

    if "elm" in sources:
        try:
            annots = fetch_elm_annotations(acc, cache_dir=cache_dir)
            all_annots.extend(annots)
            time.sleep(delay)
        except Exception as e:
            print(f"    [WARN] ELM failed for {acc}: {e}")

    if "pdb_ptm" in sources:
        try:
            annots = fetch_pdb_ptm_annotations(acc, cache_dir=cache_dir)
            all_annots.extend(annots)
            time.sleep(delay)
        except Exception as e:
            print(f"    [WARN] PDB-PTM failed for {acc}: {e}")

    return all_annots


def fetch_all_annotations_batch(
    accessions,
    sources=("uniprot", "pdbe_kb", "iptmnet", "ptmexchange", "pdb_ptm", "elm"),
    cache_dir="./annotation_cache",
    delay=0.3,
):
    """
    Fetch annotations from all sources for a list of proteins.

    Returns DataFrame with columns:
      uniprot_acc, position, aa, label, source, detail, evidence
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    all_annots = []
    for acc in tqdm(accessions, desc="Fetching annotations"):
        acc = acc.strip().upper()
        annots = fetch_all_annotations(acc, sources=sources,
                                       cache_dir=cache_dir, delay=delay)
        all_annots.extend(annots)

    df = pd.DataFrame(all_annots)
    if df.empty:
        df = pd.DataFrame(columns=[
            "uniprot_acc", "position", "aa", "label", "source", "detail", "evidence"
        ])

    # Deduplicate: same protein + position + label → keep first
    before = len(df)
    df = df.drop_duplicates(subset=["uniprot_acc", "position", "label", "source"], keep="first")
    after = len(df)
    if before > after:
        print(f"  Deduplicated: {before} → {after} annotations")

    # Add broad category
    df["category"] = df["label"].map(LABEL_CATEGORY).fillna("other")

    return df


# ═══════════════════════════════════════════════
# 6. Stratified protein sampler
# ═══════════════════════════════════════════════

# Protein family queries for diverse sampling
FAMILY_QUERIES = {
    "kinase": (
        '(ec:2.7 AND "protein kinase") AND length:[200 TO 1200]'
    ),
    "phosphatase": (
        '(ec:3.1.3 OR "phosphatase") AND length:[150 TO 1000]'
    ),
    "gpcr": (
        '"G-protein coupled receptor" AND length:[250 TO 600]'
    ),
    "enzyme_oxidoreductase": (
        "ec:1 AND length:[150 TO 800]"
    ),
    "enzyme_transferase": (
        'ec:2 NOT "protein kinase" AND length:[150 TO 800]'
    ),
    "enzyme_hydrolase": (
        'ec:3 NOT "phosphatase" AND length:[150 TO 800]'
    ),
    "transporter": (
        '"transporter" AND length:[200 TO 1000]'
    ),
    "dna_binding": (
        '("DNA-binding" OR "transcription factor") AND length:[100 TO 800]'
    ),
    "rna_binding": (
        '"RNA-binding" AND length:[100 TO 800]'
    ),
    "receptor": (
        '"receptor" NOT "G-protein coupled receptor" AND length:[200 TO 1200]'
    ),
    "immune": (
        '("immunoglobulin" OR "immune") AND length:[100 TO 800]'
    ),
    "structural": (
        '("cytoskeleton" OR "structural protein") AND length:[200 TO 1500]'
    ),
}

# Organisms with good annotation coverage
ORGANISMS = {
    "human": 9606,
    "mouse": 10090,
    "ecoli": 83333,
    "yeast": 559292,
    "arabidopsis": 3702,
    "drosophila": 7227,
    "zebrafish": 7955,
}


def search_uniprot_accessions(query, max_results=500, reviewed_only=True, annotation_score=5):
    """Search UniProt and return accession numbers."""
    import requests
    base = "https://rest.uniprot.org/uniprotkb/search"
    if reviewed_only and "reviewed:" not in query.lower():
        query = f"({query}) AND reviewed:true"
    if annotation_score is not None and "annotation_score:" not in query.lower():
        query = f"({query}) AND annotation_score:{int(annotation_score)}"

    accessions = []
    params = {"query": query, "format": "list", "size": min(max_results, 500)}
    url = base

    while url and len(accessions) < max_results:
        r = requests.get(url, params=params, timeout=30,
                         headers={"User-Agent": "RIN-v3/1.0"})
        r.raise_for_status()
        accessions.extend([a.strip() for a in r.text.strip().split("\n") if a.strip()])
        link = r.headers.get("Link", "")
        if 'rel="next"' in link:
            url = link.split(";")[0].strip(" <>")
            params = {}
        else:
            break

    return accessions[:max_results]


def sample_stratified_proteins(
    per_family=30,
    organisms=None,
    organism_weights=None,
    seed=42,
    families=None,
    min_length=100,
    max_total=500,
    annotation_score=5,
):
    """
    Build a diverse protein set by sampling across families and organisms.

    Parameters
    ----------
    per_family : int
        Target number of proteins per family.
    organisms : dict or None
        {name: taxon_id}. Default: ORGANISMS.
    organism_weights : dict or None
        {name: fraction}. Default: human 40%, mouse 15%, rest split evenly.
    seed : int
        Random seed for sampling.
    families : dict or None
        {name: query_string}. Default: FAMILY_QUERIES.
    max_total : int
        Maximum total proteins to return.
    annotation_score : int or None
        UniProt annotation score filter. Default keeps annotation_score:5.

    Returns
    -------
    list of accession strings, plus a DataFrame with sampling metadata.
    """
    rng = np.random.default_rng(seed)

    if organisms is None:
        organisms = ORGANISMS
    if families is None:
        families = FAMILY_QUERIES

    if organism_weights is None:
        # Heavier on human (best annotations), some model organisms
        organism_weights = {
            "human": 0.40,
            "mouse": 0.15,
            "ecoli": 0.10,
            "yeast": 0.10,
            "arabidopsis": 0.08,
            "drosophila": 0.08,
            "zebrafish": 0.09,
        }

    all_proteins = []
    metadata_rows = []

    for family_name, family_query in tqdm(families.items(), desc="Sampling families"):
        family_pool = []

        for org_name, org_id in organisms.items():
            weight = organism_weights.get(org_name, 0.05)
            n_target = max(1, int(per_family * weight))

            query = f"({family_query}) AND organism_id:{org_id}"
            try:
                accs = search_uniprot_accessions(
                    query,
                    max_results=n_target * 3,
                    annotation_score=annotation_score,
                )
            except Exception as e:
                print(f"    [WARN] {family_name}/{org_name}: {e}")
                continue

            # Sample from available
            if len(accs) > n_target:
                selected = list(rng.choice(accs, size=n_target, replace=False))
            else:
                selected = accs

            for acc in selected:
                family_pool.append(acc)
                metadata_rows.append({
                    "uniprot_acc": acc,
                    "family": family_name,
                    "organism": org_name,
                })

            time.sleep(0.3)  # polite

        # Cap per family
        if len(family_pool) > per_family:
            idxs = rng.choice(len(family_pool), size=per_family, replace=False)
            family_pool = [family_pool[i] for i in idxs]
            metadata_rows = [
                m for m in metadata_rows
                if m["family"] != family_name or m["uniprot_acc"] in set(family_pool)
            ]

        all_proteins.extend(family_pool)

    # Deduplicate
    seen = set()
    unique = []
    unique_meta = []
    for acc, meta in zip(all_proteins, metadata_rows):
        if acc not in seen:
            seen.add(acc)
            unique.append(acc)
            unique_meta.append(meta)

    # Cap total
    if len(unique) > max_total:
        idxs = list(rng.choice(len(unique), size=max_total, replace=False))
        unique = [unique[i] for i in idxs]
        unique_set = set(unique)
        unique_meta = [m for m in unique_meta if m["uniprot_acc"] in unique_set]

    meta_df = pd.DataFrame(unique_meta)

    print(f"\nSampled {len(unique)} proteins across {meta_df['family'].nunique()} families "
          f"and {meta_df['organism'].nunique()} organisms")
    print("\nBreakdown by family:")
    print(meta_df["family"].value_counts().to_string())
    print("\nBreakdown by organism:")
    print(meta_df["organism"].value_counts().to_string())

    return unique, meta_df


# ═══════════════════════════════════════════════
# 7. Annotation density filter
# ═══════════════════════════════════════════════


def filter_by_annotation_density(
    accessions,
    annotation_df,
    min_annotations=5,
    exclude_labels=None,
):
    """
    Keep only proteins with at least min_annotations functional annotations.
    This ensures negatives are trustworthy (well-studied proteins).

    Parameters
    ----------
    accessions : list of str
    annotation_df : DataFrame
        Output of fetch_all_annotations_batch.
    min_annotations : int
        Minimum number of training-eligible annotations per protein.
    exclude_labels : set or None
        Labels to exclude when counting (default: EXCLUDED_LABELS).

    Returns
    -------
    filtered : list of str (accessions that pass)
    removed : list of str (accessions that were dropped)
    """
    if exclude_labels is None:
        exclude_labels = EXCLUDED_LABELS

    trainable = annotation_df[~annotation_df["label"].isin(exclude_labels)]
    trainable = trainable.drop_duplicates(subset=["uniprot_acc", "position", "label"])
    counts = trainable.groupby("uniprot_acc").size()

    passed = []
    removed = []
    for acc in accessions:
        if counts.get(acc, 0) >= min_annotations:
            passed.append(acc)
        else:
            removed.append(acc)

    print(f"Annotation density filter (>={min_annotations}):")
    print(f"  Passed: {len(passed)}, Removed: {len(removed)}")

    return passed, removed


# ═══════════════════════════════════════════════
# 8. Summary statistics
# ═══════════════════════════════════════════════


def annotation_summary(annotation_df):
    """Print a nice summary of the annotation dataset."""
    print(f"Total annotations: {len(annotation_df)}")
    print(f"Proteins: {annotation_df['uniprot_acc'].nunique()}")
    print(f"Unique positions: {len(annotation_df.drop_duplicates(['uniprot_acc','position']))}")

    print("\nBy source:")
    for src, cnt in annotation_df["source"].value_counts().items():
        print(f"  {src:>12s}: {cnt}")

    print("\nBy label:")
    for lab, cnt in annotation_df["label"].value_counts().head(20).items():
        cat = LABEL_CATEGORY.get(lab, "other")
        trainable = "✓" if lab in TRAINING_LABELS else "✗"
        print(f"  {trainable} {lab:>20s}: {cnt:>5d}  ({cat})")

    print("\nBy category:")
    for cat, cnt in annotation_df["category"].value_counts().items():
        print(f"  {cat:>15s}: {cnt}")

    # Training-eligible count
    trainable = annotation_df[annotation_df["label"].isin(TRAINING_LABELS)]
    print(f"\nTraining-eligible annotations: {len(trainable)} "
          f"({len(trainable)/len(annotation_df)*100:.0f}%)")
