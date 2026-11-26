# RINMai

RINMai (Residue Interaction Network–Based Functional Mapping and Inference) produces interpretable residue-level structural-importance scores and functional-environment likelihood ratios from AlphaFold or experimental PDB structures.

## Production models

| Model | Intended structures | Training proteins | Held-out proteins | Model file |
|---|---|---:|---:|---|
| RINMai-AF50 | AlphaFold v6 | 300 | 76 | `models/af50/kde_models.pkl` |
| RINMai-PDB100 | Experimental PDB | 147 | 37 | `models/pdb100/kde_models.pkl` |

Both locked models use a Cα geometry graph with an 8 Å cutoff and class-specific Gaussian kernel-density estimators. Do not interchange AF and PDB models without an explicit cross-domain analysis.

## Outputs

For every residue, RINMai reports:

- structural importance;
- Geometry, Chemistry, Stability and Communication (GCMS) channels;
- raw class likelihood ratios (`KDELR_*`);
- a normalized seven-class spectrum (`KDE_*`);
- the highest-scoring functional environment.

The normalized spectrum is not a calibrated probability. Raw likelihood ratios are preferred for ranking residues within a class.

## Installation

RINMai requires Python 3.10 or newer and an accessible RINmaker installation.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

RINmaker must be installed separately and available on `PATH`. Its monomer-library configuration should be tested before large runs.

## Score AlphaFold structures

```bash
python src/rinmai/rin_importance_kde_lr_v2.py \
  --protein P07949 \
  --references models/af50/kde_models.pkl \
  --output predictions/af
```

For a protein list:

```bash
python src/rinmai/rin_importance_kde_lr_v2.py \
  --protein-list examples/proteins.txt \
  --references models/af50/kde_models.pkl \
  --output predictions/af
```

## Score experimental structures

```bash
python src/rinmai/rin_importance_kde_lr_pdb.py \
  --protein P07949 \
  --references models/pdb100/kde_models.pkl \
  --output predictions/pdb
```

PDB mode selects a mapped experimental structure through `pdb_selector.py`. For structure-matched benchmarking, score the exact PDB entry and chain used by the reference annotation.

## Repository layout

```text
RINMai/
├── README.md
├── requirements.txt
├── src/rinmai/                  # production implementation
├── models/
│   ├── af50/                    # AF model, manifest and locked validation
│   └── pdb100/                  # PDB model, manifest and locked validation
├── results/locked_validation/   # compact cross-model summaries
└── examples/                    # small example inputs
```

Large downloaded structures, RIN graphs, residue-level training tables and caches are intentionally excluded from the production repository. Reproducible accession manifests and compact validation outputs are retained.

## Model interpretation

RINMai is a prioritization and hypothesis-generation method. Limitations include incomplete functional annotations, absence of an explicit unknown class, monomer-only contacts for AlphaFold inputs, sensitivity of PDB analyses to conformational state, and the lack of probability calibration. It is not a clinical variant classifier.

## Reproducibility

The `sampling_metadata.csv`, `per_class_auc.csv` and `topk_accuracy.csv` files stored with each model record its source cohort and locked evaluation. Before public release, add:

- database retrieval dates;
- exact RINmaker version and configuration;
- Python environment lock file;
- code commit identifier;
- annotation-source snapshots or checksums;
- a project license and citation DOI.

## Citation

Manuscript in preparation: *RINmai: Residue Interaction Network–Based Functional Mapping and Inference*.

