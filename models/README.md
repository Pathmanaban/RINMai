# Locked model artifacts

## AF50

- Source: `af_w50/kde_models.pkl`
- Structure domain: AlphaFold v6
- Scored cohort: 376 proteins, 185,569 residues
- KDE training subset: 300 proteins
- Held-out subset: 76 proteins
- Importance weights: hub 0.25, embeddedness 0.30, chemical richness 0.225, pLDDT 0.225

## PDB100

- Source: `pdb_w100/kde_models.pkl`
- Structure domain: experimental PDB
- Scored cohort: 184 proteins, 49,374 residues
- KDE training subset: 147 proteins
- Held-out subset: 37 proteins
- Importance weights used by the active scoring code: hub 0.38, embeddedness 0.38, chemical richness 0.24

The historical PDB report contains a stale prose line stating 0.35/0.35/0.30. The executed scoring function and manuscript use 0.38/0.38/0.24.

Python pickle files must only be loaded from trusted sources. Public releases should include SHA-256 checksums and, where possible, a safer interoperable serialization format.
The current artifacts were serialized with scikit-learn 1.4.2; the production requirements pin that version to avoid estimator-compatibility warnings or numerical drift.

