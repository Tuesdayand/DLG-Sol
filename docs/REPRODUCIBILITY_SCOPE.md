# Reproducibility scope

## Current snapshot

The current pre-release snapshot supports inspection and integrity verification of the machine-readable results reported with the article. It includes complete configuration-level representation grids, aggregate paired comparisons, the six-evaluation registry, split-role label guards, fixed convex blending, regression metrics, paired RMSE bootstrap scoring, the canonical Mordred/XGBoost descriptor branch, split-safe descriptor orchestration primitives, the ChemBERTa encoder contracts, the explicit-polar-hydrogen distance-aware 3D graph branch, the canonical `aug2_head4` language–geometry branch, six evaluation-specific adapter contracts, and a provenance-checked local-package evaluation runner. The public API now enforces disjoint fit, coefficient, reserved, and label-free scored roles for each recorded evaluation; fit-only descriptor filtering; validation-selected descriptor candidates; the recorded pooled-inner ranking rule; complete fold/head prediction aggregation; ChemBERTa configuration validation; canonical inference after randomized-SMILES training augmentation; masked-mean language embedding extraction; label-free conformer graph construction; label-free 3D prediction; fit-only fusion preprocessing; label-free training-mean geometry fallback; fixed-epoch residual-head training; exact four-head neural averaging; zero- versus one-based fold seed schedules; dataset-specific winsorization; OOF, five-model, or full-refit scored-prediction aggregation; evaluation-specific coefficient reconstruction; blend-before-record aggregation; and a separate scored-label phase. Automatic benchmark acquisition and complete raw-data-to-trained-model entry points remain pending.

## Planned full release

The full release is intended to provide:

1. documented acquisition or reconstruction of each permitted benchmark dataset;
2. split and canonicalization records;
3. benchmark-specific descriptor orchestration and end-to-end XGBoost fitting;
4. complete benchmark-specific ChemBERTa fold runners and permitted checkpoint distribution;
5. dataset adapters and failure fallback for the released conformer and distance-aware 3D message-passing branch;
6. dataset-specific orchestration of the released four-head neural model and fixed prediction-level blending;
7. external-comparator adapters that are legally redistributable or independent;
8. paired bootstrap scoring;
9. regeneration of the main figures, Table 2, and machine-readable Supplementary tables.

## Evidence boundaries

External comparisons distinguish authors' released predictions, same-split reproductions, schema-and-setting adaptations, independently retrained adaptations, and pre-specified controls. The released summaries retain these evidence classes and should not be pooled as if they were equivalent replications.

Row-level data and predictions remain excluded until dataset-specific redistribution and privacy/provenance checks are complete. Pretrained weights and third-party checkpoints will not be mirrored unless their licences explicitly permit redistribution.

The historical ChemBERTa configuration recorded a pretrained model identifier but not an immutable repository revision. The released language API therefore supports protocol-level retraining and exact loading of archived compatible state dictionaries, but it does not claim that a fresh pretrained download will reproduce the historical fine-tuned weights bit for bit.

The geometry parity audit rebuilt a fixed 16-row probe under the recorded RDKit version and obtained exact node, coordinate, edge, and auxiliary tensors. Coordinate generation can nevertheless vary under other RDKit versions or platforms. Archived conformer caches and checkpoints remain hash-audited evidence rather than redistributed files.

The final fusion parity audit links all six locked neural prediction columns to their `aug2_head4` row sources and records the exact four-head seed/epoch policy. The dataset-adapter audit additionally checks 26 historical units and reproduces their partition roles, preprocessing, seed schedules, and scored-prediction aggregation. The evaluation-runner audit reconstructs all six coefficient policies from the permitted historical coefficient rows, reproduces 38,228 assembled predictions, and obtains exact metric and bootstrap parity without redistributing those rows. The earlier raw/PCA heterogeneous-member audit remains archived as non-final development evidence. Because row-level inputs and complete checkpoints are not redistributed, the public snapshot establishes configuration and prediction-provenance parity but does not claim that fresh training will reproduce historical weights bit for bit.
