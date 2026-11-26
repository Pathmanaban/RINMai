#!/usr/bin/env python3
"""
rin_importance_kde_lr_pdb.py
=============================
RINMai-PDB: Train and predict using experimental PDB structures.

Same methodology as RINMai-AF but:
  - Uses best experimental PDB structure per protein (not AlphaFold)
  - B-factor normalized and inverted as quality metric (not pLDDT)
  - Composite importance: 35% hub + 35% embed + 30% chem (no pLDDT weight)
  - Builds PDB-specific KDE reference models

Usage:
    # Full pipeline: sample → fetch PDB structures → score → train KDE → validate
    python rin_importance_kde_lr_pdb.py --full-pipeline --per-family 50 --output ./rinmai_pdb_50

    # Leave-family-out
    python rin_importance_kde_lr_pdb.py --full-pipeline --per-family 50 --leave-family-out --output ./rinmai_pdb_50_lfo

    # Score new proteins with PDB-trained models
    python rin_importance_kde_lr_pdb.py --protein P07949 --references ./rinmai_pdb_50/kde_models.pkl
"""

import argparse
import os
import sys
import json
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from rin_score_v4 import (
    score_protein,
    assemble_residue_table,
    run_rinmaker_rin,
    apply_model,
    ensure_dir,
    DEFAULT_FEATURES,
)

from pdb_selector import (
    select_best_pdb,
    download_pdb,
)

from annotation_sources import (
    fetch_all_annotations_batch,
    fetch_all_annotations,
    sample_stratified_proteins,
    filter_by_annotation_density,
    annotation_summary,
    TRAINING_LABELS,
    EXCLUDED_LABELS,
    LABEL_CATEGORY,
)


# ═══════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════

PROFILE_CLASSES = {
    "catalytic": ["ACT_SITE", "CATALYTIC"],
    "binding": ["BINDING", "LIGAND_BINDING"],
    "interface": ["INTERFACE"],
    "ptm_phospho": ["PTM_PHOSPHO", "MOD_RES"],
    "ptm_glyco_lipid": ["GLYCO", "LIPID", "PTM_GLYCO"],
    "ptm_other": ["PTM_UBIQ", "PTM_ACETYL", "PTM_METHYL", "PTM_SUMO",
                  "PTM_SNITRO", "PTM_PALMI", "PTM_MYRIS"],
    "structural": ["DISULFIDE"],
}

ALL_PROFILE_LABELS = set()
for _labels in PROFILE_CLASSES.values():
    ALL_PROFILE_LABELS.update(_labels)

PROFILE_FEATURES = [
    "Geometry_Betweenness_pct", "Geometry_Eigenvector_pct",
    "Geometry_HarmonicCloseness_pct", "Geometry_KCore_pct",
    "Geometry_Strength_pct", "Geometry_Clustering_pct",
    "ChemAll_Betweenness_pct", "ChemAll_Eigenvector_pct",
    "ChemAll_KCore_pct", "ChemAll_Clustering_pct",
    "ChemTypeRichness_pct",
    "SASA_relative", "pLDDT",
    "G", "C", "S", "M",
]


def _label_to_class(label):
    for cls, labels in PROFILE_CLASSES.items():
        if label in labels:
            return cls
    return None


# ═══════════════════════════════════════════════
# Stage 1: Structural importance (unsupervised)
# ═══════════════════════════════════════════════

def compute_structural_importance(df):
    """
    PDB-specific composite structural importance.
    No pLDDT weight — experimental structure IS ground truth.
    Weights: 35% hub + 35% embed + 30% chemistry.
    """
    hub_cols = ["Geometry_Betweenness_pct", "Geometry_Eigenvector_pct",
                "ChemAll_Betweenness_pct", "ChemAll_Eigenvector_pct"]
    embed_cols = ["Geometry_KCore_pct", "Geometry_Strength_pct",
                  "ChemAll_KCore_pct", "ChemAll_Strength_pct"]

    hub_cols = [c for c in hub_cols if c in df.columns]
    embed_cols = [c for c in embed_cols if c in df.columns]

    hub = df[hub_cols].fillna(0).mean(axis=1) if hub_cols else 0
    embed = df[embed_cols].fillna(0).mean(axis=1) if embed_cols else 0
    chem = df["ChemTypeRichness_pct"].fillna(0) if "ChemTypeRichness_pct" in df.columns else 0

    df = df.copy()
    # New (PCA-derived)
    df["StructuralImportance"] = (0.38 * hub + 0.38 * embed + 0.24 * chem).clip(0, 1)
    df["ImportanceTier"] = pd.cut(
        df["StructuralImportance"],
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["low", "moderate", "high", "very_high"],
    )
    return df


# ═══════════════════════════════════════════════
# Stage 2: KDE-LR model building
# ═══════════════════════════════════════════════

def build_kde_models(scored_ref, annot_ref, features=None):
    """
    Build one KDE per functional class + background.
    Bandwidth: Scott's rule (data-driven).
    """
    if features is None:
        features = PROFILE_FEATURES
    features = [f for f in features if f in scored_ref.columns]

    # Background statistics for standardization
    bg_data = scored_ref[features].fillna(0.0).values
    bg_mean = bg_data.mean(axis=0)
    bg_std = bg_data.std(axis=0) + 1e-9

    # Build label lookup
    annot_train = annot_ref[annot_ref["label"].isin(ALL_PROFILE_LABELS)].copy()
    label_lookup = {}
    for _, row in annot_train.iterrows():
        cls = _label_to_class(row["label"])
        if cls:
            key = (row["uniprot_acc"], row["position"])
            if key not in label_lookup:
                label_lookup[key] = set()
            label_lookup[key].add(cls)

    scored = scored_ref.copy()
    scored["_key"] = list(zip(scored["uniprot_acc"], scored["Residue"]))
    scored["_classes"] = scored["_key"].map(label_lookup)

    # Fit one KDE per class
    kde_models = {}
    class_counts = {}

    for class_name in PROFILE_CLASSES.keys():
        mask = scored["_classes"].apply(lambda x: class_name in x if isinstance(x, set) else False)
        class_data = scored.loc[mask, features].fillna(0.0).values

        if len(class_data) < 5:
            print(f"  [SKIP] {class_name}: only {len(class_data)} residues")
            continue

        X_std = (class_data - bg_mean) / bg_std
        n, d = X_std.shape
        bw = n ** (-1.0 / (d + 4))  # Scott's rule

        kde = KernelDensity(bandwidth=bw, kernel="gaussian")
        kde.fit(X_std)

        kde_models[class_name] = kde
        class_counts[class_name] = len(class_data)
        print(f"  {class_name:>16s}: {len(class_data):>5d} residues, bandwidth={bw:.4f}")

    # Background KDE
    bg_sample = scored.sample(n=min(10000, len(scored)), random_state=42)
    X_bg_std = (bg_sample[features].fillna(0.0).values - bg_mean) / bg_std
    n, d = X_bg_std.shape
    bw_bg = n ** (-1.0 / (d + 4))
    kde_bg = KernelDensity(bandwidth=bw_bg, kernel="gaussian")
    kde_bg.fit(X_bg_std)
    kde_models["background"] = kde_bg
    print(f"  {'background':>16s}: {len(bg_sample):>5d} residues, bandwidth={bw_bg:.4f}")

    return {
        "kde_models": kde_models,
        "features": features,
        "bg_mean": bg_mean,
        "bg_std": bg_std,
        "class_counts": class_counts,
    }


# ═══════════════════════════════════════════════
# Stage 2b: KDE-LR scoring
# ═══════════════════════════════════════════════

def compute_kde_lr(df, kde_data, batch_size=5000):
    """
    Compute likelihood ratio spectrum for each residue.
    LR(class) = density_class(x) / density_background(x)
    Normalized to proportions for the spectrum.
    """
    features = [f for f in kde_data["features"] if f in df.columns]
    kde_models = kde_data["kde_models"]
    bg_mean = kde_data["bg_mean"][:len(features)]
    bg_std = kde_data["bg_std"][:len(features)]

    X = df[features].fillna(0.0).values
    X_std = (X - bg_mean) / bg_std

    all_classes = [c for c in PROFILE_CLASSES.keys() if c in kde_models]
    n = len(X_std)

    # Score in batches
    log_densities = {c: np.zeros(n) for c in all_classes}
    log_bg = np.zeros(n)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = X_std[start:end]
        log_bg[start:end] = kde_models["background"].score_samples(batch)
        for c in all_classes:
            log_densities[c][start:end] = kde_models[c].score_samples(batch)

    # Likelihood ratios
    raw_lr = {}
    for c in all_classes:
        log_lr = log_densities[c] - log_bg
        raw_lr[c] = np.exp(np.clip(log_lr, -20, 20))

    # Normalize to proportions
    proportions = {c: np.zeros(n) for c in all_classes}
    for i in range(n):
        total = sum(raw_lr[c][i] for c in all_classes) + 1e-12
        for c in all_classes:
            proportions[c][i] = raw_lr[c][i] / total

    # Add to dataframe
    out = df.copy()
    for c in all_classes:
        out[f"KDE_{c}"] = np.round(proportions[c], 4)
        out[f"KDELR_{c}"] = np.round(raw_lr[c], 4)

    kde_cols = [f"KDE_{c}" for c in all_classes]
    lr_cols = [f"KDELR_{c}" for c in all_classes]

    # Best class
    out["BestClass"] = out[kde_cols].idxmax(axis=1).str.replace("KDE_", "")
    out["BestClassScore"] = out[kde_cols].max(axis=1).round(4)

    return out, kde_cols, lr_cols


# ═══════════════════════════════════════════════
# Interpretation
# ═══════════════════════════════════════════════

def interpret_residue(row):
    imp = row.get("StructuralImportance", 0)
    sasa = row.get("SASA_relative")
    state = row.get("State", "?")
    best = row.get("BestClass", "unknown")
    plddt = row.get("pLDDT", 50)

    if imp < 0.25:
        return "low structural importance"
    elif plddt is not None and not np.isnan(plddt) and plddt < 50:
        return "disordered region"
    elif sasa is not None and not np.isnan(sasa) and sasa < 0.15:
        return "buried hub (catalytic/binding pocket)"
    elif sasa is not None and not np.isnan(sasa) and sasa > 0.4 and state in ("G", "M"):
        return "surface hub (interface/signaling)"
    elif sasa is not None and not np.isnan(sasa) and sasa > 0.3:
        return "surface exposed (PTM/interaction)"
    else:
        return "structured region"


# ═══════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════

def validate_enrichment(scored_df, annot_df, col_list):
    """Test enrichment for a set of columns."""
    if not col_list:
        return pd.DataFrame()
    prefix = ""
    if col_list[0].startswith("KDELR_"):
        prefix = "KDELR_"
    elif col_list[0].startswith("KDE_"):
        prefix = "KDE_"

    results = []
    for class_name, labels in PROFILE_CLASSES.items():
        col = f"{prefix}{class_name}"
        if col not in scored_df.columns:
            continue
        sub = annot_df[annot_df["label"].isin(labels)]
        if len(sub) < 5:
            continue
        pos_keys = set(zip(sub["uniprot_acc"], sub["position"]))
        is_pos = scored_df.apply(
            lambda r: (r["uniprot_acc"], r["Residue"]) in pos_keys, axis=1)
        fg = scored_df.loc[is_pos, col].dropna()
        bg = scored_df.loc[~is_pos, col].dropna()
        if len(fg) < 5:
            continue
        stat, pval = mannwhitneyu(fg, bg, alternative="greater")
        results.append({
            "Category": class_name, "Column": col, "N": len(fg),
            "Mean_fg": round(fg.mean(), 4), "Mean_bg": round(bg.mean(), 4),
            "Fold": round(fg.mean() / max(bg.mean(), 1e-9), 2),
            "p_value": f"{pval:.2e}",
        })
    return pd.DataFrame(results)


def validate_topk(scored_df, annot_df, test_proteins, kde_cols):
    """Top-k accuracy on held-out proteins."""
    test = scored_df[scored_df["uniprot_acc"].isin(set(test_proteins))].copy()
    if test.empty:
        return pd.DataFrame()

    annot_train = annot_df[annot_df["label"].isin(ALL_PROFILE_LABELS)]
    pos_lookup = {}
    for _, row in annot_train.iterrows():
        cls = _label_to_class(row["label"])
        if cls:
            key = (row["uniprot_acc"], row["position"])
            if key not in pos_lookup:
                pos_lookup[key] = cls

    test["_key"] = list(zip(test["uniprot_acc"], test["Residue"]))
    test["_true"] = test["_key"].map(pos_lookup)
    labeled = test[test["_true"].notna()]
    if labeled.empty:
        return pd.DataFrame()

    all_classes = [c.replace("KDE_", "") for c in kde_cols]
    results = []

    for cls in labeled["_true"].unique():
        sub = labeled[labeled["_true"] == cls]
        n = len(sub)
        if n < 3:
            continue
        vals = sub[kde_cols].values
        names = np.array(all_classes)
        t1 = t2 = t3 = 0
        for i in range(n):
            ranked = names[np.argsort(-vals[i])]
            if ranked[0] == cls: t1 += 1
            if cls in ranked[:2]: t2 += 1
            if cls in ranked[:3]: t3 += 1
        results.append({
            "Class": cls, "N": n,
            "Top1_Correct": t1, "Top1_Accuracy": round(100 * t1 / n, 1),
            "Top2_Correct": t2, "Top2_Accuracy": round(100 * t2 / n, 1),
            "Top3_Correct": t3, "Top3_Accuracy": round(100 * t3 / n, 1),
        })

    df_r = pd.DataFrame(results)
    if not df_r.empty:
        total = df_r["N"].sum()
        df_r = pd.concat([df_r, pd.DataFrame([{
            "Class": "OVERALL", "N": total,
            "Top1_Correct": df_r["Top1_Correct"].sum(),
            "Top1_Accuracy": round(100 * df_r["Top1_Correct"].sum() / total, 1),
            "Top2_Correct": df_r["Top2_Correct"].sum(),
            "Top2_Accuracy": round(100 * df_r["Top2_Correct"].sum() / total, 1),
            "Top3_Correct": df_r["Top3_Correct"].sum(),
            "Top3_Accuracy": round(100 * df_r["Top3_Correct"].sum() / total, 1),
        }])], ignore_index=True)
    return df_r


def validate_importance_enrichment(scored_df, annot_df):
    """Enrichment of structural importance score vs functional categories."""
    results = []
    for cat, labels in PROFILE_CLASSES.items():
        sub = annot_df[annot_df["label"].isin(labels)]
        if len(sub) < 5:
            continue
        pos_keys = set(zip(sub["uniprot_acc"], sub["position"]))
        is_pos = scored_df.apply(
            lambda r: (r["uniprot_acc"], r["Residue"]) in pos_keys, axis=1)
        fg = scored_df.loc[is_pos, "StructuralImportance"].dropna()
        bg = scored_df.loc[~is_pos, "StructuralImportance"].dropna()
        if len(fg) < 5:
            continue
        stat, pval = mannwhitneyu(fg, bg, alternative="greater")
        results.append({
            "Category": cat, "N": len(fg),
            "Mean_fg": round(fg.mean(), 4), "Mean_bg": round(bg.mean(), 4),
            "Fold": round(fg.mean() / max(bg.mean(), 1e-9), 2),
            "p_value": f"{pval:.2e}",
        })
    return pd.DataFrame(results)


def validate_per_class_auc(scored_df, annot_df, kde_cols, test_proteins=None):
    """Compute one-vs-rest AUC for each functional class using KDE-LR proportions."""
    from sklearn.metrics import roc_auc_score

    if test_proteins is not None:
        scored = scored_df[scored_df["uniprot_acc"].isin(set(test_proteins))].copy()
    else:
        scored = scored_df.copy()

    annot_train = annot_df[annot_df["label"].isin(ALL_PROFILE_LABELS)]
    pos_lookup = {}
    for _, row in annot_train.iterrows():
        cls = _label_to_class(row["label"])
        if cls:
            key = (row["uniprot_acc"], row["position"])
            if key not in pos_lookup:
                pos_lookup[key] = cls

    scored["_key"] = list(zip(scored["uniprot_acc"], scored["Residue"]))
    scored["_true"] = scored["_key"].map(pos_lookup)

    results = []
    all_classes = [c.replace("KDE_", "") for c in kde_cols]

    for cls in all_classes:
        col = f"KDE_{cls}"
        if col not in scored.columns:
            continue
        # Binary: this class vs everything (including background)
        y_true = (scored["_true"] == cls).astype(int).values
        y_score = scored[col].fillna(0).values

        n_pos = y_true.sum()
        if n_pos < 5 or n_pos >= len(y_true) - 5:
            continue

        auc = roc_auc_score(y_true, y_score)
        results.append({
            "Class": cls,
            "N_positive": n_pos,
            "N_total": len(y_true),
            "AUC": round(auc, 3),
        })

    return pd.DataFrame(results)


def validate_precision_at_n(scored_df, annot_df, kde_cols, test_proteins=None, top_n_values=None):
    """For each class, rank all residues by KDE score and check precision at top-N."""
    if top_n_values is None:
        top_n_values = [50, 100, 200, 500]

    if test_proteins is not None:
        scored = scored_df[scored_df["uniprot_acc"].isin(set(test_proteins))].copy()
    else:
        scored = scored_df.copy()

    annot_train = annot_df[annot_df["label"].isin(ALL_PROFILE_LABELS)]
    pos_lookup = {}
    for _, row in annot_train.iterrows():
        cls = _label_to_class(row["label"])
        if cls:
            key = (row["uniprot_acc"], row["position"])
            if key not in pos_lookup:
                pos_lookup[key] = cls

    scored["_key"] = list(zip(scored["uniprot_acc"], scored["Residue"]))
    scored["_true"] = scored["_key"].map(pos_lookup)

    results = []
    for cls in [c.replace("KDE_", "") for c in kde_cols]:
        col = f"KDE_{cls}"
        if col not in scored.columns:
            continue

        sorted_df = scored.sort_values(col, ascending=False)
        n_total_pos = (sorted_df["_true"] == cls).sum()

        for n in top_n_values:
            if n > len(sorted_df):
                continue
            top_slice = sorted_df.head(n)
            true_pos = (top_slice["_true"] == cls).sum()
            precision = true_pos / n
            recall = true_pos / max(n_total_pos, 1)

            results.append({
                "Class": cls,
                "Top_N": n,
                "True_positives": true_pos,
                "Precision": round(precision, 3),
                "Recall": round(recall, 3),
                "Total_positives": n_total_pos,
            })

    return pd.DataFrame(results)


def compute_confusion_matrix(scored_df, annot_df, kde_cols, test_proteins=None):
    """Confusion matrix: for each annotated residue, what class does KDE assign?"""
    if test_proteins is not None:
        scored = scored_df[scored_df["uniprot_acc"].isin(set(test_proteins))].copy()
    else:
        scored = scored_df.copy()

    annot_train = annot_df[annot_df["label"].isin(ALL_PROFILE_LABELS)]
    pos_lookup = {}
    for _, row in annot_train.iterrows():
        cls = _label_to_class(row["label"])
        if cls:
            key = (row["uniprot_acc"], row["position"])
            if key not in pos_lookup:
                pos_lookup[key] = cls

    scored["_key"] = list(zip(scored["uniprot_acc"], scored["Residue"]))
    scored["_true"] = scored["_key"].map(pos_lookup)

    # Only keep annotated residues
    labeled = scored[scored["_true"].notna()].copy()
    if labeled.empty:
        return pd.DataFrame()

    # Predicted class = highest KDE proportion
    available = [c for c in kde_cols if c in labeled.columns]
    if not available:
        return pd.DataFrame()

    all_classes = [c.replace("KDE_", "") for c in available]
    labeled["_pred"] = labeled[available].values.argmax(axis=1)
    labeled["_pred_class"] = labeled["_pred"].map(lambda i: all_classes[i])

    # Build confusion matrix
    classes = sorted(set(labeled["_true"].unique()) | set(labeled["_pred_class"].unique()))
    cm = pd.DataFrame(0, index=classes, columns=classes)
    for _, row in labeled.iterrows():
        cm.loc[row["_true"], row["_pred_class"]] += 1

    # Normalize by row (true class)
    cm_pct = cm.div(cm.sum(axis=1), axis=0).round(3) * 100
    return cm, cm_pct


# ═══════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════

def validate_species_holdout(scored_all, annot_df, kde_cols, sampling_meta):
    """Leave-species-out: hold out each species, build KDE on rest, evaluate."""
    if sampling_meta is None or "organism" not in sampling_meta.columns:
        print("  [SKIP] No organism metadata for species holdout")
        return pd.DataFrame()

    org_map = dict(zip(sampling_meta["uniprot_acc"].str.upper(), sampling_meta["organism"]))
    scored_all_copy = scored_all.copy()
    scored_all_copy["_org"] = scored_all_copy["uniprot_acc"].map(org_map)

    all_species = sorted(scored_all_copy["_org"].dropna().unique())
    if len(all_species) < 2:
        print("  [SKIP] Need ≥2 species for holdout")
        return pd.DataFrame()

    features = [f for f in PROFILE_FEATURES if f in scored_all.columns]
    results = []

    for holdout_species in all_species:
        test_prots = set(scored_all_copy.loc[scored_all_copy["_org"] == holdout_species, "uniprot_acc"].unique())
        train_prots = set(scored_all_copy.loc[scored_all_copy["_org"] != holdout_species, "uniprot_acc"].unique())

        if len(test_prots) < 3 or len(train_prots) < 10:
            continue

        train_scored = scored_all[scored_all["uniprot_acc"].isin(train_prots)]
        train_annot = annot_df[annot_df["uniprot_acc"].isin(train_prots)]

        # Build KDE on train species
        kde_sp = build_kde_models(train_scored, train_annot, features)

        # Score test species
        test_scored = scored_all[scored_all["uniprot_acc"].isin(test_prots)].copy()
        test_scored, sp_kde_cols, _ = compute_kde_lr(test_scored, kde_sp)

        # Top-k on test
        topk = validate_topk(test_scored, annot_df, test_prots, sp_kde_cols)
        if not topk.empty:
            overall = topk[topk["Class"] == "OVERALL"]
            if not overall.empty:
                results.append({
                    "Species_held_out": holdout_species,
                    "N_test_proteins": len(test_prots),
                    "N_train_proteins": len(train_prots),
                    "Top1_Accuracy": overall["Top1_Accuracy"].values[0],
                    "Top2_Accuracy": overall["Top2_Accuracy"].values[0],
                    "Top3_Accuracy": overall["Top3_Accuracy"].values[0],
                })

    df = pd.DataFrame(results)
    if not df.empty:
        # Add overall average
        avg = pd.DataFrame([{
            "Species_held_out": "AVERAGE",
            "N_test_proteins": int(df["N_test_proteins"].mean()),
            "N_train_proteins": int(df["N_train_proteins"].mean()),
            "Top1_Accuracy": round(df["Top1_Accuracy"].mean(), 1),
            "Top2_Accuracy": round(df["Top2_Accuracy"].mean(), 1),
            "Top3_Accuracy": round(df["Top3_Accuracy"].mean(), 1),
        }])
        df = pd.concat([df, avg], ignore_index=True)
    return df


def generate_plots(scored_all, annot_df, kde_cols, output_dir, sampling_meta=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA as skPCA

    CLASS_COLORS = {
        "catalytic": "#D85A30", "binding": "#2a78d6", "interface": "#1D9E75",
        "ptm_phospho": "#7F77DD", "ptm_glyco_lipid": "#B06DD4",
        "ptm_other": "#9B59B6", "structural": "#888780",
    }

    # Build class lookup for each residue
    annot_train = annot_df[annot_df["label"].isin(ALL_PROFILE_LABELS)].copy()
    pos_class = {}
    for _, row in annot_train.iterrows():
        cls = _label_to_class(row["label"])
        if cls:
            key = (row["uniprot_acc"], row["position"])
            pos_class[key] = cls  # last wins if multi-labeled

    scored_all = scored_all.copy()
    scored_all["_func_class"] = scored_all.apply(
        lambda r: pos_class.get((r["uniprot_acc"], r["Residue"]), "background"), axis=1)

    func_mask = scored_all["_func_class"] != "background"

    # ── Figure 1: Enhanced validation (4 panels) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: SASA vs Importance colored by functional class
    ax = axes[0, 0]
    bg = scored_all[~func_mask].sample(n=min(5000, (~func_mask).sum()), random_state=42)
    ax.scatter(bg["SASA_relative"], bg["StructuralImportance"],
               c="#E1E0D9", s=4, alpha=0.15, label="Background", zorder=1, edgecolors="none")
    for cls in ["interface", "binding", "ptm_phospho", "ptm_other", "ptm_glyco_lipid", "catalytic", "structural"]:
        sub = scored_all[scored_all["_func_class"] == cls]
        if len(sub) == 0:
            continue
        ax.scatter(sub["SASA_relative"], sub["StructuralImportance"],
                   c=CLASS_COLORS.get(cls, "#888"), s=10, alpha=0.4,
                   label=f"{cls} (n={len(sub)})", zorder=3, edgecolors="none")
    ax.set_xlabel("SASA_relative (burial → exposure)")
    ax.set_ylabel("Structural Importance")
    ax.set_title("(a) Functional classes in structural space", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", markerscale=2, framealpha=0.9)
    ax.grid(alpha=0.15)

    # Panel B: PCA of KDE-LR proportions colored by functional class
    ax = axes[0, 1]
    available_kde = [c for c in kde_cols if c in scored_all.columns]
    if len(available_kde) >= 2:
        X_kde = scored_all[available_kde].fillna(0).values
        pca = skPCA(n_components=2)
        pc = pca.fit_transform(X_kde)
        var_exp = pca.explained_variance_ratio_

        bg_idx = ~func_mask
        ax.scatter(pc[bg_idx, 0], pc[bg_idx, 1], c="#E1E0D9", s=4, alpha=0.1,
                   zorder=1, edgecolors="none")
        for cls in ["interface", "binding", "ptm_phospho", "ptm_other", "ptm_glyco_lipid", "catalytic", "structural"]:
            cls_idx = scored_all["_func_class"] == cls
            if cls_idx.sum() == 0:
                continue
            ax.scatter(pc[cls_idx, 0], pc[cls_idx, 1], c=CLASS_COLORS.get(cls, "#888"),
                       s=10, alpha=0.4, label=cls, zorder=3, edgecolors="none")
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1%} variance)")
        ax.set_title("(b) KDE-LR spectrum separates functional types", fontweight="bold")
        ax.legend(fontsize=7, loc="best", markerscale=2, framealpha=0.9)
        ax.grid(alpha=0.15)

    # Panel C: SASA vs Importance colored by protein family
    ax = axes[1, 0]
    if sampling_meta is not None and "family" in sampling_meta.columns:
        fam_map = dict(zip(sampling_meta["uniprot_acc"].str.upper(), sampling_meta["family"]))
        scored_all["_family"] = scored_all["uniprot_acc"].map(fam_map)
        fam_list = sorted(scored_all["_family"].dropna().unique())
        fam_cmap = plt.cm.Set3(np.linspace(0, 1, max(len(fam_list), 1)))
        fam_colors = {f: fam_cmap[i] for i, f in enumerate(fam_list)}

        func_sub = scored_all[func_mask].copy()
        for fam in fam_list:
            sub = func_sub[func_sub["_family"] == fam]
            if len(sub) == 0:
                continue
            ax.scatter(sub["SASA_relative"], sub["StructuralImportance"],
                       c=[fam_colors[fam]], s=8, alpha=0.4, label=fam, zorder=3, edgecolors="none")
        ax.set_xlabel("SASA_relative")
        ax.set_ylabel("Structural Importance")
        ax.set_title("(c) Functional residues across protein families", fontweight="bold")
        ax.legend(fontsize=6, loc="upper right", ncol=2, markerscale=2, framealpha=0.9)
        ax.grid(alpha=0.15)
    else:
        ax.text(0.5, 0.5, "No family metadata", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(c) By protein family", fontweight="bold")

    # Panel D: Importance vs KDE-LR enrichment comparison
    ax = axes[1, 1]
    imp_enrich = validate_importance_enrichment(scored_all, annot_df)
    kde_enrich = validate_enrichment(scored_all, annot_df, kde_cols)

    if not imp_enrich.empty and not kde_enrich.empty:
        categories = imp_enrich["Category"].tolist()
        imp_folds = imp_enrich["Fold"].astype(float).tolist()
        kde_folds_map = dict(zip(kde_enrich["Category"], kde_enrich["Fold"].astype(float)))
        kde_folds = [kde_folds_map.get(c, 0) for c in categories]

        y = np.arange(len(categories))
        h = 0.35
        imp_colors = ["#D85A30" if f >= 1.0 else "#4CAF50" for f in imp_folds]
        ax.barh(y + h / 2, imp_folds, h, color=imp_colors, alpha=0.7, label="Structural importance")
        ax.barh(y - h / 2, kde_folds, h, color="#2a78d6", alpha=0.7, label="KDE-LR proportion")
        ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(categories, fontsize=9)
        ax.set_xlabel("Fold enrichment")
        ax.set_title("(d) Importance vs KDE-LR enrichment", fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="x", alpha=0.15)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "validation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved validation.png (enhanced 4-panel)")

    # ── Figure 2: PCA of KDE-LR spectrum (replaces histogram distributions) ──
    if len(available_kde) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: PCA colored by class (larger version)
        ax = axes[0]
        ax.scatter(pc[bg_idx, 0], pc[bg_idx, 1], c="#E1E0D9", s=3, alpha=0.08,
                   label="Background", zorder=1, edgecolors="none")
        for cls in ["interface", "binding", "ptm_phospho", "ptm_other", "ptm_glyco_lipid", "catalytic", "structural"]:
            cls_idx = scored_all["_func_class"] == cls
            if cls_idx.sum() == 0:
                continue
            ax.scatter(pc[cls_idx, 0], pc[cls_idx, 1], c=CLASS_COLORS.get(cls, "#888"),
                       s=14, alpha=0.5, label=f"{cls} (n={cls_idx.sum()})", zorder=3, edgecolors="none")
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1%} variance)")
        ax.set_title("KDE-LR class spectrum: PCA projection", fontweight="bold")
        ax.legend(fontsize=8, markerscale=2, framealpha=0.9)
        ax.grid(alpha=0.15)

        # Right: Per-class density contours in PCA space
        ax = axes[1]
        ax.scatter(pc[bg_idx, 0], pc[bg_idx, 1], c="#E1E0D9", s=2, alpha=0.05,
                   zorder=1, edgecolors="none")
        for cls in ["catalytic", "binding", "interface", "ptm_phospho", "structural"]:
            cls_idx = scored_all["_func_class"] == cls
            if cls_idx.sum() < 10:
                continue
            cls_pc = pc[cls_idx]
            # Draw convex hull approximation via scatter + mean marker
            ax.scatter(cls_pc[:, 0], cls_pc[:, 1], c=CLASS_COLORS.get(cls, "#888"),
                       s=8, alpha=0.3, zorder=3, edgecolors="none")
            cx, cy = cls_pc[:, 0].mean(), cls_pc[:, 1].mean()
            ax.scatter([cx], [cy], c=CLASS_COLORS.get(cls, "#888"),
                       s=120, marker="*", zorder=5, edgecolors="white", linewidth=0.5)
            ax.annotate(cls, (cx, cy), fontsize=8, fontweight="bold",
                        color=CLASS_COLORS.get(cls, "#888"),
                        xytext=(8, 8), textcoords="offset points")
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1%} variance)")
        ax.set_title("Class centroids in KDE-LR space", fontweight="bold")
        ax.grid(alpha=0.15)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "kde_lr_pca.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved kde_lr_pca.png")

    # Clean up temp columns
    scored_all.drop(columns=["_func_class"], inplace=True, errors="ignore")
    scored_all.drop(columns=["_family"], inplace=True, errors="ignore")


def generate_classification_plots(scored_all, annot_df, kde_cols, output_dir,
                                   test_proteins=None):
    """Generate per-class AUC, precision@N, and confusion matrix plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    CLASS_COLORS = {
        "catalytic": "#D85A30", "binding": "#2a78d6", "interface": "#1D9E75",
        "ptm_phospho": "#7F77DD", "ptm_glyco_lipid": "#B06DD4",
        "ptm_other": "#9B59B6", "structural": "#888780",
    }

    # Prepare labels
    annot_train = annot_df[annot_df["label"].isin(ALL_PROFILE_LABELS)]
    pos_lookup = {}
    for _, row in annot_train.iterrows():
        cls = _label_to_class(row["label"])
        if cls:
            key = (row["uniprot_acc"], row["position"])
            if key not in pos_lookup:
                pos_lookup[key] = cls

    if test_proteins is not None:
        scored = scored_all[scored_all["uniprot_acc"].isin(set(test_proteins))].copy()
    else:
        scored = scored_all.copy()

    scored["_key"] = list(zip(scored["uniprot_acc"], scored["Residue"]))
    scored["_true"] = scored["_key"].map(pos_lookup)

    all_classes = [c.replace("KDE_", "") for c in kde_cols if c in scored.columns]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── Panel A: ROC curves per class ──
    ax = axes[0]
    auc_results = validate_per_class_auc(scored_all, annot_df, kde_cols, test_proteins)
    for _, row in auc_results.iterrows():
        cls = row["Class"]
        col = f"KDE_{cls}"
        if col not in scored.columns:
            continue
        y_true = (scored["_true"] == cls).astype(int).values
        y_score = scored[col].fillna(0).values
        if y_true.sum() < 5:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, color=CLASS_COLORS.get(cls, "#888"),
                label=f"{cls} (AUC={row['AUC']:.2f}, n={row['N_positive']})", linewidth=1.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("(a) Per-class ROC curves", fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.15)

    # ── Panel B: Confusion matrix heatmap ──
    ax = axes[1]
    result = compute_confusion_matrix(scored_all, annot_df, kde_cols, test_proteins)
    if result is not None and len(result) == 2:
        cm_raw, cm_pct = result
        classes_order = [c for c in ["catalytic", "binding", "interface", "ptm_phospho",
                                      "ptm_other", "ptm_glyco_lipid", "structural"]
                         if c in cm_pct.index]
        cm_plot = cm_pct.loc[classes_order, classes_order] if len(classes_order) > 0 else cm_pct

        im = ax.imshow(cm_plot.values, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(cm_plot.columns)))
        ax.set_xticklabels([c.replace("ptm_", "p_") for c in cm_plot.columns],
                           rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(cm_plot.index)))
        ax.set_yticklabels([c.replace("ptm_", "p_") for c in cm_plot.index], fontsize=8)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title("(b) Confusion matrix (%)", fontweight="bold")

        # Add percentage text in cells
        for i in range(len(cm_plot.index)):
            for j in range(len(cm_plot.columns)):
                val = cm_plot.iloc[i, j]
                color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold" if i == j else "normal")

        plt.colorbar(im, ax=ax, shrink=0.8, label="% of true class")
    else:
        ax.text(0.5, 0.5, "No confusion data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(b) Confusion matrix", fontweight="bold")

    # ── Panel C: Per-class AUC bar chart ──
    ax = axes[2]
    if not auc_results.empty:
        auc_sorted = auc_results.sort_values("AUC", ascending=True)
        y = np.arange(len(auc_sorted))
        colors_auc = [CLASS_COLORS.get(c, "#888") for c in auc_sorted["Class"]]
        ax.barh(y, auc_sorted["AUC"], color=colors_auc, alpha=0.8, height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(auc_sorted["Class"], fontsize=9)
        ax.set_xlabel("AUC (one-vs-rest)")
        ax.set_xlim(0.5, 1.0)
        ax.axvline(0.5, color="black", linestyle="--", alpha=0.3)
        ax.set_title("(c) Per-class AUC", fontweight="bold")
        ax.grid(axis="x", alpha=0.15)
        for i, (_, row) in enumerate(auc_sorted.iterrows()):
            ax.text(row["AUC"] + 0.005, i, f"{row['AUC']:.3f}", va="center",
                    fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classification_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved classification_metrics.png")


# ═══════════════════════════════════════════════
# Score a new protein
# ═══════════════════════════════════════════════

def score_protein_full(acc, kde_data, out_dir="./predictions",
                       cache_dir="./annotation_cache"):
    """Score all residues in a protein using PDB structure + KDE-LR spectrum."""
    acc = acc.strip().upper()

    # 1. Score from PDB
    df = score_protein_pdb_mode(acc,
                                out_dir=out_dir,
                                rin_dir=os.path.join(out_dir, "rin"),
                                pdb_cache=os.path.join(out_dir, "pdb_cache"))

    # 2. Structural importance (PDB weights)
    df = compute_structural_importance(df)

    # 3. KDE-LR spectrum
    df, kde_cols, lr_cols = compute_kde_lr(df, kde_data)

    # 4. Annotations
    raw = fetch_all_annotations(acc, sources=("uniprot", "pdbe_kb", "iptmnet"),
                                cache_dir=cache_dir)
    annots_df = pd.DataFrame(raw) if raw else pd.DataFrame()

    if not annots_df.empty:
        label_agg = annots_df.groupby("position").agg({
            "label": lambda x: ";".join(sorted(set(x))),
            "source": lambda x: ";".join(sorted(set(x))),
            "detail": lambda x: ";".join(sorted(set(str(d) for d in x if d)))[:200],
        }).reset_index()
        label_agg.columns = ["Residue", "KnownLabels", "AnnotSources", "AnnotDetail"]
        df = df.merge(label_agg, on="Residue", how="left")
    else:
        df["KnownLabels"] = None
        df["AnnotSources"] = None
        df["AnnotDetail"] = None

    # 5. Interpretation
    df["Interpretation"] = df.apply(interpret_residue, axis=1)

    # 6. Output columns
    out_cols = [
        "Residue", "Resname", "AA", "pLDDT", "SASA_relative",
        "G", "C", "S", "M", "State", "MaxGCMS",
        "StructuralImportance", "ImportanceTier", "Interpretation",
        "BestClass", "BestClassScore",
    ] + sorted(kde_cols) + sorted(lr_cols) + [
        "KnownLabels", "AnnotSources", "AnnotDetail",
    ]
    out_cols = [c for c in out_cols if c in df.columns]

    # Summary
    n_total = len(df)
    n_high = (df["StructuralImportance"] >= 0.75).sum()
    n_annotated = df["KnownLabels"].notna().sum()

    print(f"\n  {'─' * 50}")
    print(f"  {acc} — {n_total} residues scored")
    print(f"  {'─' * 50}")
    print(f"  Very high importance (≥0.75): {n_high} ({100*n_high/n_total:.1f}%)")
    print(f"  Known annotations: {n_annotated}")
    print(f"  Best class distribution:")
    if "BestClass" in df.columns:
        for cls, cnt in df["BestClass"].value_counts().items():
            print(f"    {cls:>16s}: {cnt:>4d} ({100*cnt/n_total:.1f}%)")

    print(f"\n  Top 10 residues:")
    print(df[out_cols].sort_values("StructuralImportance", ascending=False).head(10).round(3).to_string(index=False))

    return df[out_cols].round(4)


def normalize_bfactor(df):
    """Convert B-factor to pLDDT-like quality (0-100, high=rigid=good)."""
    df = df.copy()
    if "pLDDT" not in df.columns:
        df["pLDDT"] = 90.0
        return df

    bfactors = df["pLDDT"].values.copy()
    df["Bfactor_raw"] = bfactors

    b_min = np.nanmin(bfactors)
    b_max = np.nanmax(bfactors)
    if b_max - b_min > 0.01:
        b_norm = (bfactors - b_min) / (b_max - b_min)
    else:
        b_norm = np.zeros_like(bfactors)

    df["pLDDT"] = np.clip((1.0 - b_norm) * 100.0, 0, 100)
    return df


def score_protein_pdb_mode(acc, out_dir, rin_dir, pdb_cache="./pdb_cache"):
    """Score a protein using its best experimental PDB structure."""
    info = select_best_pdb(acc, cache_dir=pdb_cache)
    if info is None:
        raise RuntimeError(f"No PDB structure for {acc}")

    pdb_id = info["pdb_id"]
    chain_id = info["chain_id"]

    struct_dir = os.path.join(out_dir, "structures")
    ensure_dir(struct_dir)
    ensure_dir(rin_dir)

    # Download
    pdb_path = download_pdb(pdb_id, out_dir=struct_dir)

    # RINmaker
    graphml_base = os.path.join(rin_dir, f"{pdb_id}__{chain_id}")
    graphml_path = graphml_base + ".graphml"
    if not os.path.exists(graphml_path):
        # run_rinmaker_rin may append .graphml itself
        run_rinmaker_rin(pdb_path, graphml_base)
        # Check which file was created
        if not os.path.exists(graphml_path) and os.path.exists(graphml_base + ".graphml.graphml"):
            os.rename(graphml_base + ".graphml.graphml", graphml_path)
        elif not os.path.exists(graphml_path) and os.path.exists(graphml_base):
            os.rename(graphml_base, graphml_path)

    # Assemble features
    df = assemble_residue_table(pdb_path, chain_id, "CA", 8.0, graphml_path, rin_dir)
    df["uniprot_acc"] = acc

    # Map to UniProt numbering
    auth_start = info.get("auth_start", info["unp_start"])
    offset = info["unp_start"] - int(auth_start)
    df["PDB_Residue"] = df["Residue"].copy()
    df["Residue"] = df["Residue"] + offset
    df = df[(df["Residue"] >= info["unp_start"]) &
            (df["Residue"] <= info["unp_end"])].copy()

    # Normalize B-factor
    df = normalize_bfactor(df)

    res_str = f"{info['resolution']:.1f}Å" if info['resolution'] else "N/A"
    print(f"  PDB: {pdb_id} chain {chain_id} ({info['method']}, {res_str}, "
          f"{len(df)} residues, UniProt {info['unp_start']}-{info['unp_end']})")

    return df


# ═══════════════════════════════════════════════
# Full pipeline
# ═══════════════════════════════════════════════

def run_full_pipeline(per_family=50, output_dir="./kde_lr_output",
                      leave_family_out=False, equal_species=False, seed=42):
    """Full pipeline: sample → annotate → score → build KDE → validate → report."""
    ensure_dir(output_dir)

    # Species weights
    if equal_species:
        organism_weights = {
            "human": 1/7, "mouse": 1/7, "ecoli": 1/7, "yeast": 1/7,
            "arabidopsis": 1/7, "drosophila": 1/7, "zebrafish": 1/7,
        }
        print("  Species weighting: EQUAL (14% each)")
    else:
        organism_weights = None  # use default (40% human)
        print("  Species weighting: DEFAULT (40% human)")

    # 1. Sample
    print("=" * 60)
    print("[1/6] Sampling proteins")
    print("=" * 60)
    protein_ids, sampling_meta = sample_stratified_proteins(
        per_family=per_family, seed=seed, organism_weights=organism_weights)

    # 2. Annotations
    print(f"\n[2/6] Fetching annotations for {len(protein_ids)} proteins")
    annot_df = fetch_all_annotations_batch(
        protein_ids, sources=("uniprot", "pdbe_kb", "iptmnet"),
        cache_dir="./annotation_cache")
    annotation_summary(annot_df)
    filtered_ids, _ = filter_by_annotation_density(protein_ids, annot_df, min_annotations=5)

    # 3. Score (RIN + SASA from PDB structures)
    print(f"\n[3/6] Scoring {len(filtered_ids)} proteins (PDB structures)")
    all_scored = []
    failed = []
    for acc in tqdm(filtered_ids, desc="Scoring (PDB)"):
        try:
            df = score_protein_pdb_mode(acc,
                               out_dir=output_dir,
                               rin_dir=os.path.join(output_dir, "rin"),
                               pdb_cache=os.path.join(output_dir, "pdb_cache"))
            all_scored.append(df)
        except Exception as e:
            failed.append((acc, str(e)))
    if not all_scored:
        raise RuntimeError("No proteins scored")
    scored_all = pd.concat(all_scored, ignore_index=True)
    print(f"  {len(scored_all)} residues across {len(all_scored)} proteins")
    if failed:
        print(f"  Failed: {len(failed)}")
        for acc, err in failed[:5]:
            print(f"    {acc}: {err}")

    # 4. Structural importance
    print(f"\n[4/6] Computing structural importance")
    scored_all = compute_structural_importance(scored_all)

    # 5. Build KDE + score
    print(f"\n[5/6] Building KDE-LR models")

    # Train/test split
    rng = np.random.default_rng(seed)

    if leave_family_out and "family" in sampling_meta.columns:
        # Split by FAMILY — hold out entire families
        family_map = dict(zip(sampling_meta["uniprot_acc"].str.upper(), sampling_meta["family"]))
        all_families = list(set(family_map.values()))
        rng.shuffle(all_families)
        n_test_families = max(1, len(all_families) // 5)  # ~20% of families
        test_families = set(all_families[:n_test_families])
        train_families = set(all_families[n_test_families:])

        scored_all_accs = set(scored_all["uniprot_acc"].unique())
        train_proteins = {p for p in scored_all_accs if family_map.get(p, "UNKNOWN") in train_families}
        test_proteins = {p for p in scored_all_accs if family_map.get(p, "UNKNOWN") in test_families}

        print(f"  Leave-family-out: train families={len(train_families)}, test families={len(test_families)}")
        print(f"  Test families: {test_families}")
    else:
        # Split by protein (leave-protein-out)
        all_proteins = scored_all["uniprot_acc"].unique().tolist()
        rng.shuffle(all_proteins)
        split = int(0.8 * len(all_proteins))
        train_proteins = set(all_proteins[:split])
        test_proteins = set(all_proteins[split:])

    print(f"  Train: {len(train_proteins)}, Test: {len(test_proteins)}")

    train_scored = scored_all[scored_all["uniprot_acc"].isin(train_proteins)]
    train_annot = annot_df[annot_df["uniprot_acc"].isin(train_proteins)]

    kde_data = build_kde_models(train_scored, train_annot)

    # Score all residues
    scored_all, kde_cols, lr_cols = compute_kde_lr(scored_all, kde_data)

    # 6. Validate + report
    print(f"\n[6/6] Validation")

    imp_enrich = validate_importance_enrichment(scored_all, annot_df)
    print("\nStructural importance enrichment:")
    print(imp_enrich.to_string(index=False))

    kde_enrich = validate_enrichment(scored_all, annot_df, kde_cols)
    print("\nKDE-LR enrichment (normalized proportions):")
    print(kde_enrich.to_string(index=False))

    lr_enrich = validate_enrichment(scored_all, annot_df, lr_cols)
    print("\nKDE-LR enrichment (raw likelihood ratios):")
    print(lr_enrich.to_string(index=False))

    topk = validate_topk(scored_all, annot_df, test_proteins, kde_cols)
    print("\nTop-k accuracy (held-out proteins):")
    print(topk.to_string(index=False))

    # Per-class AUC
    per_class_auc = validate_per_class_auc(scored_all, annot_df, kde_cols, test_proteins)
    print("\nPer-class AUC (one-vs-rest, held-out proteins):")
    print(per_class_auc.to_string(index=False))

    # Precision at top-N
    prec_at_n = validate_precision_at_n(scored_all, annot_df, kde_cols, test_proteins)
    print("\nPrecision at top-N (held-out proteins):")
    print(prec_at_n.to_string(index=False))

    # Confusion matrix
    cm_result = compute_confusion_matrix(scored_all, annot_df, kde_cols, test_proteins)
    if cm_result is not None and len(cm_result) == 2:
        cm_raw, cm_pct = cm_result
        print("\nConfusion matrix (% of true class):")
        print(cm_pct.to_string())

    # Species holdout validation (always runs)
    print("\nSpecies holdout validation:")
    species_holdout = validate_species_holdout(scored_all, annot_df, kde_cols, sampling_meta)
    if not species_holdout.empty:
        print(species_holdout.to_string(index=False))

    # Save everything
    print(f"\nSaving to {output_dir}/")
    with open(os.path.join(output_dir, "kde_models.pkl"), "wb") as f:
        pickle.dump(kde_data, f)
    scored_all.to_csv(os.path.join(output_dir, "all_scored.tsv"), sep="\t", index=False)
    annot_df.to_csv(os.path.join(output_dir, "annotations.csv"), index=False)
    sampling_meta.to_csv(os.path.join(output_dir, "sampling_metadata.csv"), index=False)
    imp_enrich.to_csv(os.path.join(output_dir, "enrichment_importance.csv"), index=False)
    kde_enrich.to_csv(os.path.join(output_dir, "enrichment_kde.csv"), index=False)
    lr_enrich.to_csv(os.path.join(output_dir, "enrichment_lr.csv"), index=False)
    topk.to_csv(os.path.join(output_dir, "topk_accuracy.csv"), index=False)
    if not species_holdout.empty:
        species_holdout.to_csv(os.path.join(output_dir, "species_holdout.csv"), index=False)
    per_class_auc.to_csv(os.path.join(output_dir, "per_class_auc.csv"), index=False)
    prec_at_n.to_csv(os.path.join(output_dir, "precision_at_n.csv"), index=False)
    if cm_result is not None and len(cm_result) == 2:
        cm_raw.to_csv(os.path.join(output_dir, "confusion_matrix_raw.csv"))
        cm_pct.to_csv(os.path.join(output_dir, "confusion_matrix_pct.csv"))

    generate_plots(scored_all, annot_df, kde_cols, output_dir, sampling_meta=sampling_meta)
    generate_classification_plots(scored_all, annot_df, kde_cols, output_dir,
                                   test_proteins=test_proteins)

    # Report
    lines = [
        "RINMai-PDB: Structural Importance + KDE-LR Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Method: KDE likelihood ratios on EXPERIMENTAL PDB structures",
        f"Importance weights: 35% hub + 35% embed + 30% chem (no pLDDT)",
        f"Proteins: {scored_all['uniprot_acc'].nunique()}",
        f"Residues: {len(scored_all)}",
        f"Train: {len(train_proteins)}, Test: {len(test_proteins)}",
        f"Leave-family-out: {leave_family_out}",
        f"Species weighting: {'equal' if equal_species else 'default (40% human)'}",
        "",
        "KDE models:",
    ]
    for cls, cnt in kde_data["class_counts"].items():
        lines.append(f"  {cls:>16s}: {cnt}")
    lines.extend([
        "",
        "Structural importance enrichment:",
        imp_enrich.to_string(index=False),
        "",
        "KDE-LR enrichment (proportions):",
        kde_enrich.to_string(index=False),
        "",
        "KDE-LR enrichment (raw likelihood ratios):",
        lr_enrich.to_string(index=False),
        "",
        "Top-k accuracy (held-out proteins):",
        topk.to_string(index=False),
        "",
        "Per-class AUC (one-vs-rest):",
        per_class_auc.to_string(index=False),
        "",
        "Precision at top-N:",
        prec_at_n.to_string(index=False),
    ])
    if cm_result is not None and len(cm_result) == 2:
        lines.extend([
            "",
            "Confusion matrix (% of true class):",
            cm_pct.to_string(),
        ])
    if not species_holdout.empty:
        lines.extend([
            "",
            "Species holdout validation (leave-one-species-out):",
            species_holdout.to_string(index=False),
        ])

    with open(os.path.join(output_dir, "report.txt"), "w") as f:
        f.write("\n".join(lines))
    print("  Saved report.txt")

    print(f"\n  Done. All outputs in {output_dir}/")
    return scored_all, annot_df, kde_data


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RIN-based residue importance + KDE likelihood ratio scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build references + validate
  python rin_importance_kde_lr.py --full-pipeline --per-family 50 --output ./results

  # Leave-family-out validation
  python rin_importance_kde_lr.py --full-pipeline --per-family 50 --leave-family-out --output ./results_lfo

  # Score new proteins
  python rin_importance_kde_lr.py --protein P07949 --references ./results/kde_models.pkl
  python rin_importance_kde_lr.py --protein-list proteins.txt --references ./results/kde_models.pkl
        """,
    )
    parser.add_argument("--protein", "-p", type=str, default=None)
    parser.add_argument("--protein-list", "-l", type=str, default=None)
    parser.add_argument("--references", "-r", type=str, default=None,
                        help="Path to kde_models.pkl")
    parser.add_argument("--full-pipeline", action="store_true")
    parser.add_argument("--per-family", type=int, default=50)
    parser.add_argument("--leave-family-out", action="store_true")
    parser.add_argument("--equal-species", action="store_true",
                        help="Equal weight per species instead of human-heavy (default: 40%% human)")
    parser.add_argument("--output", "-o", type=str, default="./kde_lr_output")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.full_pipeline:
        run_full_pipeline(
            per_family=args.per_family,
            output_dir=args.output,
            leave_family_out=args.leave_family_out,
            equal_species=args.equal_species,
            seed=args.seed,
        )

    elif args.protein or args.protein_list:
        if not args.references:
            print("Error: --references required. Run --full-pipeline first.")
            sys.exit(1)

        with open(args.references, "rb") as f:
            kde_data = pickle.load(f)

        proteins = []
        if args.protein:
            proteins = [p.strip() for p in args.protein.split(",") if p.strip()]
        elif args.protein_list:
            with open(args.protein_list) as f:
                proteins = [l.strip() for l in f if l.strip() and not l.startswith("#")]

        ensure_dir(args.output)
        all_results = []

        for acc in proteins:
            print(f"\n{'=' * 60}")
            print(f"  Scoring {acc}")
            print(f"{'=' * 60}")
            try:
                result = score_protein_full(acc, kde_data, out_dir=args.output)
                out_file = os.path.join(args.output, f"{acc}_scored.tsv")
                result.to_csv(out_file, sep="\t", index=False)
                print(f"\n  Saved → {out_file}")
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()

        if len(all_results) > 1:
            combined = pd.concat(all_results, ignore_index=True)
            combined.to_csv(os.path.join(args.output, "combined_scored.tsv"), sep="\t", index=False)
            print(f"\nCombined: {len(combined)} residues → {args.output}/combined_scored.tsv")

        print(f"\nDone. {len(all_results)}/{len(proteins)} proteins scored.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
