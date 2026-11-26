"""
rin_score_v2.py
================
RIN-based residue importance scoring — v2.

Changes from v1:
  - SASA (solvent-accessible surface area) added as a key feature
  - LLR / frozen-prior scoring track REMOVED
  - Single learned model: P(functional | RIN features + SASA + pLDDT)
  - DISULFIDE excluded from training positives (structurally trivial)
  - Empirical priors kept as descriptive analytics only
  - GCMS channels kept as interpretive labels, not scoring inputs

Architecture:
  Structure → RIN + Geometry → Network metrics + SASA → Feature vector
                                                            ↓
                                                    Learned model → P(functional)

Requires:
    numpy, pandas, networkx, MDAnalysis, scipy, scikit-learn, tqdm, requests
    Optional: freesasa (faster SASA; falls back to built-in Shrake-Rupley)
    External: RINmaker binary in PATH
"""

# ═══════════════════════════════════════════════
# PART 0: Imports & utilities
# ═══════════════════════════════════════════════

import os
import re
import time
import math
import json
import subprocess
import urllib.request
from urllib.error import HTTPError, URLError
from typing import List, Tuple, Optional, Dict, Iterable
from dataclasses import dataclass, asdict
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import networkx as nx
import MDAnalysis as mda
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ═══════════════════════════════════════════════
# PART 1: Structure fetchers (AlphaFold & PDB)
# ═══════════════════════════════════════════════

AFDB_URLS = {
    "pdb": "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.pdb",
    "cif": "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.cif",
}
RCSB_URLS = {
    "cif": "https://files.rcsb.org/download/{pdbid}.cif",
    "pdb": "https://files.rcsb.org/download/{pdbid}.pdb",
}


def download_alphafold_model(uniprot_acc, out_dir, prefer="pdb", retries=2):
    acc = uniprot_acc.strip().upper()
    ensure_dir(out_dir)
    dest = os.path.join(out_dir, f"AF-{acc}-F1-model_v6.{prefer}")
    if os.path.exists(dest):
        return dest
    url = AFDB_URLS[prefer].format(acc=acc)
    for attempt in range(retries + 1):
        try:
            print(f"  Downloading AlphaFold v6 {acc}")
            req = urllib.request.Request(url, headers={"User-Agent": "RIN-v2"})
            with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as w:
                w.write(r.read())
            return dest
        except HTTPError as e:
            if e.code == 404:
                print(f"  AlphaFold {prefer} for {acc} not found.")
                return None
        except URLError:
            time.sleep(1.0 * (attempt + 1))
    return None


def download_pdb_structure(pdbid, out_dir, prefer="cif", retries=2):
    pdbid = pdbid.strip().lower()
    ensure_dir(out_dir)
    dest = os.path.join(out_dir, f"{pdbid}.{prefer}")
    if os.path.exists(dest):
        return dest
    url = RCSB_URLS[prefer].format(pdbid=pdbid)
    for attempt in range(retries + 1):
        try:
            print(f"  Downloading PDB {pdbid}")
            req = urllib.request.Request(url, headers={"User-Agent": "RIN-v2"})
            with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as w:
                w.write(r.read())
            return dest
        except HTTPError as e:
            if e.code == 404:
                print(f"  PDB {pdbid} not found.")
                return None
        except URLError:
            time.sleep(1.0 * (attempt + 1))
    return None


_uniprot_pat = re.compile(
    r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$"
)
_pdb_pat = re.compile(r"^[0-9][A-Za-z0-9]{3}$")


def fetch_structure(identifier, out_dir="./cache", prefer_af="pdb",
                    prefer_pdb="cif", chain_default="A"):
    ident = identifier.strip()
    if _pdb_pat.match(ident):
        path = download_pdb_structure(ident, out_dir=out_dir, prefer=prefer_pdb)
        return path, chain_default
    if _uniprot_pat.match(ident.upper()) or (len(ident) in (6, 10) and ident[0].isalpha()):
        path = download_alphafold_model(ident, out_dir=out_dir, prefer=prefer_af)
        return path, chain_default
    print("  Unknown identifier type.")
    return None, chain_default


# ═══════════════════════════════════════════════
# PART 2: SASA computation
# ═══════════════════════════════════════════════

# Van der Waals radii (Å) — standard set
VDW_RADII = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20,
    "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "SE": 1.90, "FE": 1.47, "ZN": 1.39, "MG": 1.73, "CA": 1.74,
    "MN": 1.61, "CU": 1.40, "CO": 1.52, "NI": 1.63,
}

PROBE_RADIUS = 1.4  # water probe


def _generate_sphere_points(n=92):
    """Golden-spiral sphere points for Shrake-Rupley."""
    pts = np.zeros((n, 3))
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        y = 1 - (2.0 * i) / (n - 1)
        radius_at_y = np.sqrt(1 - y * y)
        theta = golden_angle * i
        pts[i] = [radius_at_y * np.cos(theta), y, radius_at_y * np.sin(theta)]
    return pts


def compute_sasa_shrake_rupley(positions, elements, n_points=92):
    """
    Pure-numpy Shrake-Rupley SASA.

    Parameters
    ----------
    positions : (N, 3) array of atom coordinates
    elements : list of element symbols (length N)
    n_points : number of sphere test points per atom

    Returns
    -------
    sasa : (N,) array of per-atom SASA in Å²
    """
    n_atoms = len(positions)
    sphere_pts = _generate_sphere_points(n_points)

    radii = np.array([
        VDW_RADII.get(e.upper().strip(), 1.70) + PROBE_RADIUS
        for e in elements
    ])

    tree = KDTree(positions)
    max_radius = radii.max()

    sasa = np.zeros(n_atoms)
    area_per_point = 4.0 * np.pi / n_points

    for i in range(n_atoms):
        r_i = radii[i]
        test_pts = positions[i] + r_i * sphere_pts  # (n_points, 3)

        # find neighbors within sum of max radii
        neighbor_idxs = tree.query_ball_point(positions[i], r_i + max_radius)

        n_accessible = 0
        for pt in test_pts:
            buried = False
            for j in neighbor_idxs:
                if j == i:
                    continue
                dist_sq = np.sum((pt - positions[j]) ** 2)
                if dist_sq < radii[j] ** 2:
                    buried = True
                    break
            if not buried:
                n_accessible += 1

        sasa[i] = n_accessible * area_per_point * r_i * r_i

    return sasa


def compute_residue_sasa(struct_path, chain_id="A", n_points=92):
    """
    Compute per-residue SASA.

    Tries freesasa first (fast C library), falls back to built-in Shrake-Rupley.

    Returns DataFrame with columns: Chain, Residue, SASA, SASA_relative
    """
    u = mda.Universe(struct_path)
    sel = u.select_atoms(f"(segid {chain_id} or chainID {chain_id}) and protein")
    if sel.n_atoms == 0:
        sel = u.select_atoms("protein")
    if sel.n_atoms == 0:
        raise ValueError("No protein atoms found.")

    # Try freesasa first
    try:
        import freesasa
        struct = freesasa.Structure(struct_path)
        result = freesasa.calc(struct)
        # aggregate per residue
        area_classes = freesasa.classifyResults(result, struct)
        # freesasa gives total/polar/apolar per residue
        # Use the selections API for per-residue
        residue_sasa = {}
        for i in range(struct.nAtoms()):
            resi = struct.residueNumber(i).strip()
            chain = struct.chainLabel(i).strip() or "A"
            key = (chain, resi)
            if key not in residue_sasa:
                residue_sasa[key] = 0.0
            residue_sasa[key] += result.atomArea(i)

        rows = []
        for (ch, resi), area in residue_sasa.items():
            try:
                rows.append({"Chain": ch, "Residue": int(resi), "SASA": area})
            except ValueError:
                continue
        df = pd.DataFrame(rows)
        if not df.empty:
            df["SASA_pct"] = df["SASA"].rank(pct=True).fillna(0.0)
        print("  SASA computed via freesasa")
        return df

    except ImportError:
        pass

    # Fallback: built-in Shrake-Rupley
    print("  SASA via built-in Shrake-Rupley (freesasa not installed — consider pip install freesasa)")
    positions = sel.positions
    elements = []
    for atom in sel.atoms:
        elem = getattr(atom, "element", None)
        if not elem or elem.strip() == "":
            # infer from atom name
            name = atom.name.strip()
            elem = name[0] if name else "C"
        elements.append(elem.strip())

    atom_sasa = compute_sasa_shrake_rupley(positions, elements, n_points=n_points)

    # Aggregate per residue
    residue_areas = {}
    for atom, area in zip(sel.atoms, atom_sasa):
        chain = str(atom.segid).strip() or str(getattr(atom, "chainID", "")).strip() or "A"
        resi = int(atom.resid)
        key = (chain, resi)
        residue_areas[key] = residue_areas.get(key, 0.0) + area

    rows = [{"Chain": ch, "Residue": resi, "SASA": area}
            for (ch, resi), area in residue_areas.items()]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["SASA_pct"] = df["SASA"].rank(pct=True).fillna(0.0)
    return df


# Max SASA per residue type (Tien et al. 2013, theoretical)
# Used to compute relative SASA
MAX_SASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}


def add_relative_sasa(df):
    """Add SASA_relative = SASA / max_SASA_for_residue_type."""
    if "Resname" in df.columns and "SASA" in df.columns:
        df["SASA_relative"] = df.apply(
            lambda r: r["SASA"] / MAX_SASA.get(str(r["Resname"]).upper(), 200.0)
            if pd.notna(r["SASA"]) else np.nan,
            axis=1,
        )
        df["SASA_relative"] = df["SASA_relative"].clip(0.0, 1.5)  # cap at 1.5
    return df


# ═══════════════════════════════════════════════
# PART 3: RINmaker + GraphML parser + chemistry graphs
# ═══════════════════════════════════════════════


def run_rinmaker_rin(struct_path, out_graphml):
    ensure_dir(os.path.dirname(out_graphml) or ".")
    cmd = [
        "RINmaker", "-i", struct_path, "-o", out_graphml, "rin",
        "--policy", "multiple",
        "--hydrogen-bond", "3.5", "--vdw-bond", "0.5",
        "--ionic-bond", "4.0", "--pication-bond", "5.0",
        "--pipistack-bond", "6.5", "--h-bond-angle", "63",
        "--pication-angle", "45",
        "--pipistack-normal-normal", "30", "--pipistack-normal-centre", "60",
    ]
    print("  >>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _graphml_ns(root):
    return {"g": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}

def _q(ns, tag):
    return f"{{{ns['g']}}}{tag}" if ns else tag

def _graphml_keymap(root, ns):
    key_map = {}
    for k in root.findall(_q(ns, "key")):
        kid = k.attrib.get("id")
        name = k.attrib.get("attr.name")
        if kid and name:
            key_map[kid] = name
    return key_map


def parse_rin_graphml_edges(graphml_path):
    tree = ET.parse(graphml_path)
    root = tree.getroot()
    ns = _graphml_ns(root)
    key_map = _graphml_keymap(root, ns)

    def parse_node_id(nid):
        parts = str(nid).split(":")
        if len(parts) >= 2:
            ch = parts[0].strip() or "A"
            try:
                pos = int(parts[1])
            except Exception:
                pos = None
            return ch, pos
        return "A", None

    rows = []
    for e in root.findall(f".//{_q(ns, 'edge')}"):
        src, tgt = e.attrib.get("source"), e.attrib.get("target")
        if not src or not tgt:
            continue
        c1, r1 = parse_node_id(src)
        c2, r2 = parse_node_id(tgt)
        if r1 is None or r2 is None:
            continue

        data = {}
        for d in e.findall(_q(ns, "data")):
            kid = d.attrib.get("key")
            data[key_map.get(kid, kid)] = d.text

        raw_inter = str(data.get("Interaction", "")).strip()
        head = raw_inter.split(":")[0].upper() if raw_inter else ""
        subtype = raw_inter.split(":", 1)[1] if ":" in raw_inter else None
        etype_map = {
            "HBOND": "Hbond", "VDW": "VdW", "IONIC": "Ionic",
            "PICATION": "PiCation", "PIPISTACK": "PiPi", "HYDROPHOBIC": "VdW",
        }
        etype = etype_map.get(head, head.lower() if head else "chem")

        dist = data.get("Distance")
        energy = data.get("Energy")
        try:
            dist = float(dist) if dist is not None else np.nan
        except Exception:
            dist = np.nan
        try:
            energy = float(energy) if energy is not None else np.nan
        except Exception:
            energy = np.nan

        rows.append({
            "Chain1": c1, "Res1": int(r1), "Chain2": c2, "Res2": int(r2),
            "Type": etype, "InteractionRaw": raw_inter, "Subtype": subtype,
            "Distance": dist, "Energy": energy,
        })
    return pd.DataFrame(rows)


def build_chemistry_graphs(edge_df):
    graphs = {}
    if edge_df is None or edge_df.empty:
        graphs["ChemAll"] = nx.Graph(layer="chemistry:All")
        return graphs

    for t, sub in edge_df.groupby("Type"):
        G = nx.Graph(layer=f"chemistry:{t}")
        for _, r in sub.iterrows():
            u = (str(r["Chain1"]) or "A", int(r["Res1"]))
            v = (str(r["Chain2"]) or "A", int(r["Res2"]))
            d = r.get("Distance", np.nan)
            try:
                d = float(d)
            except Exception:
                d = np.nan
            w = 1.0 / max(d, 1e-6) if np.isfinite(d) and d > 0 else 1.0
            attrs = {"weight": w, "distance": d, "resistance": 1.0 / max(w, 1e-9), "type": t}
            if G.has_edge(u, v):
                if w > G[u][v].get("weight", 0):
                    G[u][v].update(attrs)
            else:
                G.add_edge(u, v, **attrs)
        graphs[t] = G

    Gall = nx.Graph(layer="chemistry:All")
    for t, Gt in graphs.items():
        for u, v, a in Gt.edges(data=True):
            w = a.get("weight", 1.0)
            if Gall.has_edge(u, v):
                if w > Gall[u][v].get("weight", 0):
                    Gall[u][v].update(a)
            else:
                Gall.add_edge(u, v, **a)
    graphs["ChemAll"] = Gall
    return graphs


# ═══════════════════════════════════════════════
# PART 4: Geometry graph + network metrics
# ═══════════════════════════════════════════════


def load_chain(u, chain_id):
    sel = u.select_atoms(f"(segid {chain_id} or chainID {chain_id}) and protein")
    if sel.n_atoms == 0:
        sel = u.select_atoms("protein")
    if sel.n_atoms == 0:
        raise ValueError(f"No protein atoms for chain '{chain_id}'.")
    return sel


def residue_coords(selection, atom_name="CA", cb_fallback_to_ca=True):
    coords, metas = [], []
    for res in selection.residues:
        if atom_name.upper() == "CB":
            atm = res.atoms.select_atoms("name CB")
            if atm.n_atoms == 0 and cb_fallback_to_ca:
                atm = res.atoms.select_atoms("name CA")
        else:
            atm = res.atoms.select_atoms("name CA")
        if atm.n_atoms == 0:
            continue
        xyz = atm.positions[0]
        try:
            plddt = float(atm.tempfactors[0])
        except Exception:
            plddt = np.nan
        chain = str(res.segid).strip() or str(getattr(res, "chainID", "")).strip() or "A"
        metas.append({
            "Chain": chain, "Residue": int(res.resid),
            "Resname": str(res.resname), "pLDDT": plddt,
        })
        coords.append(xyz)
    if not coords:
        raise ValueError("No residue coordinates found.")
    return np.asarray(coords, float), pd.DataFrame(metas)


def build_geometry_graph(selection, cutoff=8.0, atom_name="CA"):
    coords, meta = residue_coords(selection, atom_name=atom_name, cb_fallback_to_ca=True)
    meta["Chain"] = meta["Chain"].astype(str).replace({"": "A", "nan": "A"}).fillna("A")
    nodes = [(meta.loc[i, "Chain"], int(meta.loc[i, "Residue"])) for i in range(len(meta))]
    tree = KDTree(coords)

    G = nx.Graph(layer=f"geometry:{atom_name}_cut{cutoff}")
    for n, row in zip(nodes, meta.to_dict("records")):
        G.add_node(n, **row)

    for i in range(len(nodes)):
        for j in tree.query_ball_point(coords[i], cutoff):
            if j <= i:
                continue
            d = float(np.linalg.norm(coords[i] - coords[j]))
            w = 1.0 / max(d, 1e-6)
            G.add_edge(nodes[i], nodes[j], weight=w, distance=d, resistance=1.0 / max(w, 1e-9))
    return G, meta


def compute_network_metrics(G, prefix):
    if G is None or G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["Chain", "Residue"])

    nodes = list(G.nodes())
    deg = dict(G.degree())
    wdeg = dict(G.degree(weight="weight"))

    try:
        clust = nx.clustering(G, weight="weight")
    except Exception:
        clust = {n: 0.0 for n in nodes}

    if nx.get_edge_attributes(G, "resistance"):
        btw = nx.betweenness_centrality(G, weight="resistance", normalized=True)
        harm = nx.harmonic_centrality(G, distance="resistance")
    else:
        btw = nx.betweenness_centrality(G, normalized=True)
        harm = nx.harmonic_centrality(G)

    try:
        eig = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eig = nx.pagerank(G, weight="weight")

    try:
        kcore = nx.core_number(G)
    except Exception:
        kcore = {n: 0 for n in nodes}

    rows = []
    for n in nodes:
        chain, res = n
        rows.append({
            "Chain": chain, "Residue": int(res),
            f"{prefix}_Degree": deg.get(n, 0),
            f"{prefix}_Strength": wdeg.get(n, 0.0),
            f"{prefix}_Clustering": clust.get(n, 0.0),
            f"{prefix}_Betweenness": btw.get(n, 0.0),
            f"{prefix}_Eigenvector": eig.get(n, 0.0),
            f"{prefix}_HarmonicCloseness": harm.get(n, 0.0),
            f"{prefix}_KCore": kcore.get(n, 0),
        })
    df = pd.DataFrame(rows)
    for c in [c for c in df.columns if c.startswith(prefix + "_")]:
        df[c + "_pct"] = df[c].rank(pct=True).fillna(0.0)
    return df


# ═══════════════════════════════════════════════
# PART 5: GCMS channel scores (descriptive only)
# ═══════════════════════════════════════════════


def mean_ignore_na(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series([0.0] * len(df), index=df.index)
    return df[cols].astype(float).mean(axis=1, skipna=True).fillna(0.0)


def add_chem_type_richness(df):
    out = df.copy()
    typed_deg = [
        c for c in out.columns
        if c.endswith("_Degree")
        and not c.startswith("Geometry_")
        and not c.startswith("ChemAll_")
    ]
    if not typed_deg:
        out["ChemTypeRichness"] = 0
        out["ChemTypeRichness_pct"] = 0.0
        return out
    out["ChemTypeRichness"] = (out[typed_deg].fillna(0) > 0).sum(axis=1)
    out["ChemTypeRichness_pct"] = out["ChemTypeRichness"].rank(pct=True).fillna(0.0)
    return out


def build_channel_scores(wide):
    """Compute G, C, S, M channels. These are DESCRIPTIVE labels, not scoring inputs."""
    df = add_chem_type_richness(wide.copy())

    geom_comm = ["Geometry_Betweenness_pct", "Geometry_HarmonicCloseness_pct", "Geometry_Eigenvector_pct"]
    geom_stab = ["Geometry_KCore_pct", "Geometry_Strength_pct", "Geometry_Clustering_pct"]
    chem_comm = ["ChemAll_Betweenness_pct", "ChemAll_HarmonicCloseness_pct", "ChemAll_Eigenvector_pct"]
    chem_stab = ["ChemAll_KCore_pct", "ChemAll_Strength_pct", "ChemAll_Clustering_pct"]

    df["G"] = mean_ignore_na(df, geom_comm + geom_stab)

    typed_str = [
        c for c in [
            "VdW_Strength_pct", "Hbond_Strength_pct", "Ionic_Strength_pct",
            "PiCation_Strength_pct", "PiPi_Strength_pct",
        ]
        if c in df.columns
    ]
    df["C"] = mean_ignore_na(df, typed_str + ["ChemTypeRichness_pct"])
    df["M"] = mean_ignore_na(df, geom_comm + chem_comm)
    df["S"] = mean_ignore_na(df, geom_stab + chem_stab)

    for col in ["G", "C", "S", "M"]:
        df[col] = df[col].clip(0.0, 1.0)

    # Dominant state label
    STATE_ORDER = ["G", "C", "S", "M"]
    def _assign_state(row):
        vals = {k: float(row.get(k, 0.0) if pd.notna(row.get(k, 0.0)) else 0.0) for k in STATE_ORDER}
        if all(v <= 1e-9 for v in vals.values()):
            return None
        return max(STATE_ORDER, key=lambda k: (vals[k], -STATE_ORDER.index(k)))

    df["State"] = df.apply(_assign_state, axis=1)
    df["MaxGCMS"] = df[["G", "C", "S", "M"]].astype(float).max(axis=1).fillna(0.0).clip(0.0, 1.0)

    return df


# ═══════════════════════════════════════════════
# PART 6: Assemble wide table (metrics + SASA)
# ═══════════════════════════════════════════════

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "SEC": "U", "PYL": "O",
}


def assemble_residue_table(struct_path, chain_id, atom, cutoff,
                           rin_graphml_path, rin_dir):
    """Build the full wide table: geometry + chemistry metrics + SASA + GCMS channels."""
    u = mda.Universe(struct_path)
    sel = load_chain(u, chain_id)

    # Geometry graph + metrics
    Gg, meta = build_geometry_graph(sel, cutoff=cutoff, atom_name=atom)
    geom = compute_network_metrics(Gg, prefix="Geometry")

    base = meta.copy()
    base["Chain"] = base["Chain"].astype(str).replace({"": "A", "nan": "A"}).fillna("A")
    base["Residue"] = base["Residue"].astype(int)
    out = base.merge(geom, on=["Chain", "Residue"], how="left")

    # Chemistry graphs + metrics
    edges = parse_rin_graphml_edges(rin_graphml_path)
    chem_graphs = build_chemistry_graphs(edges)
    for lname, Gc in chem_graphs.items():
        cdf = compute_network_metrics(Gc, prefix=lname)
        out = out.merge(cdf, on=["Chain", "Residue"], how="left")

    # SASA
    try:
        sasa_df = compute_residue_sasa(struct_path, chain_id=chain_id)
        out = out.merge(sasa_df, on=["Chain", "Residue"], how="left")
        out = add_relative_sasa(out)
    except Exception as e:
        print(f"  [WARN] SASA computation failed: {e}")
        out["SASA"] = np.nan
        out["SASA_pct"] = np.nan
        out["SASA_relative"] = np.nan

    # GCMS channels (descriptive)
    out = build_channel_scores(out)

    # Single-letter AA
    out["AA"] = out["Resname"].astype(str).str.upper().map(AA3TO1).fillna("X")

    # Metadata
    out["SourceModel"] = os.path.basename(struct_path)
    out["ChainQueried"] = chain_id
    out["Atom"] = atom
    out["CutoffA"] = float(cutoff)

    # Save geometry graph
    nx.write_graphml(
        Gg,
        os.path.join(rin_dir, f"{os.path.basename(struct_path)}__{chain_id}.geometry.graphml"),
    )
    return out


# ═══════════════════════════════════════════════
# PART 7: Score one protein (end-to-end)
# ═══════════════════════════════════════════════


def score_protein(identifier, out_dir, rin_dir, chain="A", atom="CA", cutoff=8.0):
    struct_path, chain_id = fetch_structure(identifier, out_dir=out_dir, chain_default=chain)
    if struct_path is None:
        raise FileNotFoundError(f"Could not fetch structure for {identifier}")

    os.makedirs(rin_dir, exist_ok=True)

    graphml_base = os.path.join(rin_dir, f"{os.path.basename(struct_path)}__{chain_id}")
    graphml_path = graphml_base + ".graphml"

    if not os.path.exists(graphml_path):
        run_rinmaker_rin(struct_path, graphml_base)
    else:
        print(f"  Using cached GraphML: {graphml_path}")

    wide = assemble_residue_table(
        struct_path, chain_id, atom, cutoff, graphml_path, rin_dir,
    )
    return wide


# ═══════════════════════════════════════════════
# PART 8: UniProt annotation fetcher
# ═══════════════════════════════════════════════

FEATURE_TYPE_MAP = {
    "Active site": "ACT_SITE",
    "Binding site": "BINDING",
    "Site": "SITE",
    "Modified residue": "MOD_RES",
    "Glycosylation": "GLYCO",
    "Lipidation": "LIPID",
    "Cross-link": "CROSSLINK",
    "Disulfide bond": "DISULFIDE",
}

PTM_CLASS_BY_AA = {
    "S": "PTM_STY", "T": "PTM_STY", "Y": "PTM_STY",
    "N": "PTM_N", "R": "PTM_R", "K": "PTM_K", "C": "PTM_C", "M": "PTM_M",
}

# Labels that count as "functional" positives for training
# DISULFIDE excluded — structurally trivial, dominates training
TRAINING_POSITIVE_LABELS = {"ACT_SITE", "BINDING", "SITE", "MOD_RES", "GLYCO", "LIPID"}


@dataclass
class AnnotatedResidue:
    uniprot_acc: str
    position: int
    aa: str
    label: str
    ptm_class: Optional[str] = None
    description: str = ""
    evidence: str = ""


def fetch_uniprot_entry(acc, retries=3, delay=0.5):
    import requests
    url = f"https://rest.uniprot.org/uniprotkb/{acc.strip()}?format=json"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "RIN-v2/1.0"})
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                print(f"  [WARN] {acc} not found")
                return None
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 2)))
                continue
        except Exception as e:
            print(f"  [ERR] {acc} attempt {attempt}: {e}")
            time.sleep(delay * (attempt + 1))
    return None


def parse_annotations(entry):
    acc = entry.get("primaryAccession", "?")
    seq = entry.get("sequence", {}).get("value", "")
    results = []
    for feat in entry.get("features", []):
        label = FEATURE_TYPE_MAP.get(feat.get("type", ""))
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
        for pos in range(int(start), int(end) + 1):
            if pos < 1 or pos > len(seq):
                continue
            aa = seq[pos - 1]
            ptm_class = PTM_CLASS_BY_AA.get(aa, f"PTM_{aa}") if label in (
                "MOD_RES", "GLYCO", "LIPID", "CROSSLINK") else None
            results.append(AnnotatedResidue(
                acc, pos, aa, label, ptm_class, desc, ";".join(ecos)))
    return results


def fetch_annotations_batch(accessions, cache_dir="./uniprot_cache", delay=0.3):
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    all_annots = []
    for acc in tqdm(accessions, desc="Fetching UniProt"):
        acc = acc.strip().upper()
        cached = os.path.join(cache_dir, f"{acc}.json") if cache_dir else None
        if cached and os.path.exists(cached):
            with open(cached) as f:
                entry = json.load(f)
        else:
            entry = fetch_uniprot_entry(acc)
            if entry is None:
                continue
            if cached:
                with open(cached, "w") as f:
                    json.dump(entry, f)
            time.sleep(delay)
        for a in parse_annotations(entry):
            all_annots.append(asdict(a))
    df = pd.DataFrame(all_annots)
    if df.empty:
        df = pd.DataFrame(columns=[
            "uniprot_acc", "position", "aa", "label", "ptm_class", "description", "evidence"
        ])
    return df


def search_uniprot_accessions(query, max_results=500, reviewed_only=True):
    import requests
    base = "https://rest.uniprot.org/uniprotkb/search"
    if reviewed_only and "reviewed:" not in query.lower():
        query = f"({query}) AND reviewed:true"
    accessions, params = [], {"query": query, "format": "list", "size": min(max_results, 500)}
    url = base
    while url and len(accessions) < max_results:
        r = requests.get(url, params=params, timeout=30,
                         headers={"User-Agent": "RIN-v2/1.0"})
        r.raise_for_status()
        accessions.extend([a.strip() for a in r.text.strip().split("\n") if a.strip()])
        link = r.headers.get("Link", "")
        if 'rel="next"' in link:
            url = link.split(";")[0].strip(" <>")
            params = {}
        else:
            break
    return accessions[:max_results]


# ═══════════════════════════════════════════════
# PART 9: Empirical priors (ANALYTICS ONLY)
# ═══════════════════════════════════════════════


def compute_empirical_priors(scored_df, annotation_df, states=None,
                             min_count=30, pseudocount=1.0):
    """
    Compute state distributions per annotation class.
    For interpretation and papers — NOT used in scoring.
    Works with both old format (ptm_class column) and new format (label-based).
    """
    if states is None:
        states = ["G", "C", "S", "M"]

    # Handle both old format (position column) and new format
    right_col = "position" if "position" in annotation_df.columns else "Residue"
    merged = scored_df.merge(
        annotation_df,
        left_on=["uniprot_acc", "Residue"],
        right_on=["uniprot_acc", right_col],
        how="inner",
    )

    priors = {}

    # General (all residues)
    gs = scored_df["State"].dropna()
    if len(gs) >= min_count:
        counts = {s: pseudocount for s in states}
        for s in gs:
            if s in counts:
                counts[s] += 1
        total = sum(counts.values())
        priors["General"] = {s: round(counts[s] / total, 4) for s in states}

    # Per annotation label — auto-detect all labels present
    all_labels = sorted(merged["label"].dropna().unique())
    for label in all_labels:
        sub = merged.loc[merged["label"] == label, "State"].dropna()
        if len(sub) < min_count:
            print(f"  [SKIP] {label}: {len(sub)} residues (need {min_count})")
            continue
        counts = {s: pseudocount for s in states}
        for s in sub:
            if s in counts:
                counts[s] += 1
        total = sum(counts.values())
        priors[label] = {s: round(counts[s] / total, 4) for s in states}

    return priors


def priors_to_dataframe(priors):
    rows = []
    for label, dist in priors.items():
        row = {"label": label}
        row.update(dist)
        rows.append(row)
    return pd.DataFrame(rows).set_index("label")


# ═══════════════════════════════════════════════
# PART 10: Build training set + learn model
# ═══════════════════════════════════════════════

# Default features for the model
DEFAULT_FEATURES = [
    # GCMS channels
    "G", "C", "S", "M", "MaxGCMS",
    # Geometry network metrics — RAW (absolute, cross-protein comparable)
    "Geometry_Betweenness", "Geometry_Eigenvector",
    "Geometry_HarmonicCloseness", "Geometry_KCore",
    "Geometry_Strength", "Geometry_Clustering",
    # Geometry network metrics — PERCENTILE (within-protein rank)
    "Geometry_Betweenness_pct", "Geometry_Eigenvector_pct",
    "Geometry_HarmonicCloseness_pct", "Geometry_KCore_pct",
    "Geometry_Strength_pct", "Geometry_Clustering_pct",
    # Chemistry network metrics — RAW
    "ChemAll_Betweenness", "ChemAll_Eigenvector",
    "ChemAll_HarmonicCloseness", "ChemAll_KCore",
    "ChemAll_Strength", "ChemAll_Clustering",
    # Chemistry network metrics — PERCENTILE
    "ChemAll_Betweenness_pct", "ChemAll_Eigenvector_pct",
    "ChemAll_HarmonicCloseness_pct", "ChemAll_KCore_pct",
    "ChemAll_Strength_pct", "ChemAll_Clustering_pct",
    # Typed chemistry — RAW
    "VdW_Strength", "Hbond_Strength", "Ionic_Strength",
    "PiCation_Strength", "PiPi_Strength",
    # Typed chemistry — PERCENTILE
    "VdW_Strength_pct", "Hbond_Strength_pct", "Ionic_Strength_pct",
    "PiCation_Strength_pct", "PiPi_Strength_pct",
    # Chemistry richness
    "ChemTypeRichness", "ChemTypeRichness_pct",
    # Surface accessibility
    "SASA", "SASA_relative",
    # Structure confidence
    "pLDDT",
    # Graph-level context (protein size proxy)
    "Geometry_Degree", "ChemAll_Degree",
]


def build_training_set(scored_df, annotation_df,
                       positive_labels=None,
                       negative_strategy="sampled",
                       neg_ratio=3.0,
                       class_weights=None,
                       seed=42):
    """
    Build binary classification dataset.

    Parameters
    ----------
    positive_labels : set or None
        Which annotation labels count as positives.
        Default: TRAINING_POSITIVE_LABELS (excludes DISULFIDE).
    class_weights : dict or None
        Optional per-label weight for positives, e.g. {"ACT_SITE": 3.0, "MOD_RES": 1.0}.
        Applied as sample_weight column.
    """
    if positive_labels is None:
        positive_labels = TRAINING_POSITIVE_LABELS

    # Filter annotations to positive labels only
    pos_annot = annotation_df[annotation_df["label"].isin(positive_labels)]
    pos_keys = set(zip(pos_annot["uniprot_acc"], pos_annot["position"]))

    # Also build label lookup for class weighting
    label_lookup = {}
    for _, row in pos_annot.iterrows():
        key = (row["uniprot_acc"], row["position"])
        if key not in label_lookup:
            label_lookup[key] = row["label"]

    scored = scored_df.copy()
    scored["_key"] = list(zip(scored["uniprot_acc"], scored["Residue"]))
    scored["y"] = scored["_key"].apply(lambda k: 1 if k in pos_keys else 0)

    # Annotation label for positives (for diagnostics)
    scored["_annot_label"] = scored["_key"].map(label_lookup)

    # Sample weight
    if class_weights:
        scored["sample_weight"] = scored.apply(
            lambda r: class_weights.get(r["_annot_label"], 1.0) if r["y"] == 1 else 1.0,
            axis=1,
        )
    else:
        scored["sample_weight"] = 1.0

    scored.drop(columns=["_key"], inplace=True)

    positives = scored[scored["y"] == 1]
    negatives = scored[scored["y"] == 0]

    # Also exclude DISULFIDE residues from negatives (they're neither positive nor typical background)
    disulfide_keys = set(zip(
        annotation_df.loc[annotation_df["label"] == "DISULFIDE", "uniprot_acc"],
        annotation_df.loc[annotation_df["label"] == "DISULFIDE", "position"],
    ))
    neg_key_col = list(zip(negatives["uniprot_acc"], negatives["Residue"]))
    negatives = negatives[[k not in disulfide_keys for k in neg_key_col]]

    if negative_strategy == "sampled":
        n_neg = int(len(positives) * neg_ratio)
        if n_neg < len(negatives):
            negatives = negatives.sample(n=n_neg, random_state=seed)

    dataset = pd.concat([positives, negatives], ignore_index=True)
    dataset = dataset.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"Training set: {len(positives)} positives, {len(negatives)} negatives")
    if "_annot_label" in dataset.columns:
        print("  Positive breakdown:")
        for lab, cnt in positives["_annot_label"].value_counts().items():
            print(f"    {lab:>12s}: {cnt}")

    return dataset


def learn_model(dataset, feature_cols=None, cv_folds=5, seed=42):
    """
    Train a gradient boosted classifier (HistGradientBoosting).
    Uses protein-level CV (GroupKFold) — entire proteins are held out, not individual residues.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, GroupKFold
    from sklearn.metrics import classification_report

    if feature_cols is None:
        feature_cols = [c for c in DEFAULT_FEATURES if c in dataset.columns]

    # Safety: filter to columns that actually exist
    feature_cols = [c for c in feature_cols if c in dataset.columns]
    print(f"Features ({len(feature_cols)}):")
    for f in feature_cols:
        print(f"  {f}")

    X = dataset[feature_cols].values  # HistGB handles NaN natively
    y = dataset["y"].values

    # Protein-level groups for CV
    if "uniprot_acc" in dataset.columns:
        groups = dataset["uniprot_acc"].values
        n_unique = len(set(groups))
        actual_folds = min(cv_folds, n_unique)
        cv = GroupKFold(n_splits=actual_folds)
        cv_type = f"protein-level GroupKFold ({actual_folds} folds, {n_unique} proteins)"
    else:
        from sklearn.model_selection import StratifiedKFold
        groups = None
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        cv_type = f"residue-level StratifiedKFold ({cv_folds} folds)"
    print(f"  CV strategy: {cv_type}")

    # Class imbalance: compute sample weights
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    scale = n_neg / n_pos if n_pos > 0 else 1.0
    sw = np.where(y == 1, scale, 1.0)

    model = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=1.0,
        max_bins=255,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=seed,
    )

    # Cross-validated AUC
    cv_aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc",
                               groups=groups,
                               fit_params={"sample_weight": sw})

    # Fit on full data
    model.fit(X, y, sample_weight=sw)

    # Classification report
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, target_names=["Background", "Functional"])

    print(f"\nCV AUC: {cv_aucs.mean():.3f} +/- {cv_aucs.std():.3f}")
    print(report)

    # Feature importances — try built-in, fallback to permutation
    try:
        raw_imp = model.feature_importances_
        imp_type = "built-in"
    except AttributeError:
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(model, X, y, n_repeats=10,
                                       random_state=seed, scoring="roc_auc")
        raw_imp = perm.importances_mean
        imp_type = "permutation"

    importances = dict(zip(feature_cols, raw_imp))
    sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    print(f"Feature importances ({imp_type}):")
    for feat, imp in sorted_imp.items():
        bar = "#" * int(imp * 200)
        print(f"  {feat:>35s} : {imp:.4f}  {bar}")

    return {
        "model": model,
        "feature_cols": feature_cols,
        "importances": sorted_imp,
        "cv_auc_mean": float(cv_aucs.mean()),
        "cv_auc_std": float(cv_aucs.std()),
        "cv_report": report,
    }


def apply_model(scored_df, model_result):
    """Add LearnedScore (probability) and LearnedLogit to the dataframe."""
    model = model_result["model"]
    feature_cols = model_result["feature_cols"]
    df = scored_df.copy()
    # Add missing feature columns as NaN (HistGB handles NaN natively)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    X = df[feature_cols].values
    df["LearnedScore"] = model.predict_proba(X)[:, 1]
    try:
        df["LearnedLogit"] = model.decision_function(X)
    except AttributeError:
        df["LearnedLogit"] = np.log(
            df["LearnedScore"].clip(1e-9, 1 - 1e-9) / (1 - df["LearnedScore"].clip(1e-9, 1 - 1e-9))
        )
    return df


# ═══════════════════════════════════════════════
# PART 11: Full pipeline
# ═══════════════════════════════════════════════


def run_full_pipeline(
    protein_ids,
    out_root="./v2_output",
    chain="A",
    atom="CA",
    cutoff=8.0,
    uniprot_cache="./uniprot_cache",
    min_prior_count=20,
    neg_ratio=3.0,
    class_weights=None,
    cv_folds=5,
    seed=42,
):
    """
    End-to-end v2 pipeline.

    Returns dict with:
      annotations, scored_all, empirical_priors, model_result
    """
    os.makedirs(out_root, exist_ok=True)

    # ── Step 1: Annotations ──
    print("=" * 60)
    print("STEP 1: Fetching UniProt annotations")
    print("=" * 60)
    annot_df = fetch_annotations_batch(protein_ids, cache_dir=uniprot_cache)
    n_func = len(annot_df[annot_df["label"].isin(TRAINING_POSITIVE_LABELS)])
    n_dss = len(annot_df[annot_df["label"] == "DISULFIDE"])
    print(f"  {len(annot_df)} total annotations ({n_func} functional, {n_dss} disulfide)")

    # ── Step 2: Score proteins ──
    print("\n" + "=" * 60)
    print("STEP 2: Computing RIN metrics + SASA")
    print("=" * 60)
    all_scored = []
    for acc in tqdm(protein_ids, desc="Scoring"):
        try:
            df = score_protein(
                acc,
                out_dir=os.path.join(out_root, "cache"),
                rin_dir=os.path.join(out_root, "rin"),
                chain=chain, atom=atom, cutoff=cutoff,
            )
            df["uniprot_acc"] = acc
            all_scored.append(df)
        except Exception as e:
            print(f"  [ERR] {acc}: {e}")

    if not all_scored:
        raise RuntimeError("No proteins scored successfully.")

    scored_all = pd.concat(all_scored, ignore_index=True)
    print(f"  {len(scored_all)} residues across {len(all_scored)} proteins")

    # Check SASA coverage
    sasa_ok = scored_all["SASA"].notna().sum()
    print(f"  SASA computed for {sasa_ok}/{len(scored_all)} residues ({100*sasa_ok/len(scored_all):.0f}%)")

    # ── Step 3: Empirical priors (analytics) ──
    print("\n" + "=" * 60)
    print("STEP 3: Empirical priors (descriptive analytics)")
    print("=" * 60)
    emp_priors = compute_empirical_priors(
        scored_all, annot_df, min_count=min_prior_count,
    )
    print("\n" + priors_to_dataframe(emp_priors).round(3).to_string())
    with open(os.path.join(out_root, "empirical_priors.json"), "w") as f:
        json.dump(emp_priors, f, indent=2)

    # ── Step 4: Learn model ──
    print("\n" + "=" * 60)
    print("STEP 4: Training model (DISULFIDE excluded from positives)")
    print("=" * 60)
    train_df = build_training_set(
        scored_all, annot_df,
        negative_strategy="sampled",
        neg_ratio=neg_ratio,
        class_weights=class_weights,
        seed=seed,
    )
    model_result = learn_model(train_df, cv_folds=cv_folds, seed=seed)

    # ── Step 5: Apply model ──
    scored_all = apply_model(scored_all, model_result)

    # Save
    scored_all.to_csv(os.path.join(out_root, "all_scored.tsv"), sep="\t", index=False)

    import pickle
    with open(os.path.join(out_root, "model.pkl"), "wb") as f:
        pickle.dump(model_result, f)

    return {
        "annotations": annot_df,
        "scored_all": scored_all,
        "empirical_priors": emp_priors,
        "model_result": model_result,
    }
