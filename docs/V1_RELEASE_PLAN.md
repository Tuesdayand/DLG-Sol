# v1.0 reproducibility-package plan

## Scope

Version 1.0 will be a user-supplied-data reproducibility package. It will provide an executable path from lawfully obtained benchmark inputs through DLG-Sol training, prediction, fixed blending, scoring, and regeneration of public article outputs. It will not bundle raw third-party benchmark rows, third-party source trees, or model weights without clear redistribution permission.

Fresh training under the recorded protocol is a protocol reproduction. Version 1.0 will not claim bitwise reproduction of historical weights when the original pretrained revision, row-level inputs, or checkpoints cannot legally or technically be redistributed.

## Gate V1-A: input provenance and reconstruction

- complete the dataset-specific redistribution decision for every primary and supplementary source;
- document lawful acquisition or reconstruction where permitted;
- define canonical local-package schemas for structures, labels, partitions, and record identifiers;
- preserve source revision, retrieval date, original hash, transformation history, and split assignment;
- audit the provenance of every file required by an external-comparator adapter.

## Gate V1-B: end-to-end DLG-Sol workflow

- connect benchmark input validation and canonicalization to descriptor fitting;
- run benchmark-specific ChemBERTa fine-tuning and embedding extraction;
- run conformer construction, 3D-MPNN fitting, and geometry embedding extraction;
- apply dataset-specific preprocessing and failed-geometry fallback;
- fit and aggregate the four residual fusion heads;
- reconstruct the fold-, split-, or evaluation-specific fixed blend coefficient without scored-label leakage;
- generate final component and DLG-Sol predictions for all six primary evaluations.

The command-line workflow must accept only explicit local inputs, must not download unlicensed resources, and must keep scored labels outside preprocessing, fitting, and coefficient selection.

## Gate V1-C: result regeneration

- reproduce RMSE, MAE, and paired-bootstrap comparisons from generated predictions;
- regenerate Table 2, the chemical-subgroup figure, and all machine-readable Supplementary tables supported by public inputs;
- record expected numerical tolerances for fresh training separately from exact deterministic transformations;
- execute full and reduced smoke tests in pinned environments.

## Gate V1-D: archival release

- update `CITATION.cff` to version 1.0.0 and the final release date;
- update README, data-access documentation, release checklist, and reproducibility scope from development to released status;
- make the validator emit `PUBLIC_RELEASE_READY=YES` only when all v1.0 gates pass;
- create and push a signed or annotated `v1.0.0` tag;
- create a GitHub Release and archive it with a persistent DOI;
- add the archival DOI to citation metadata and the manuscript availability statement.

## Definition of ready

Version 1.0 is ready when a user with lawful access to the required inputs can follow the public documentation and execute the supported workflow without private paths, undisclosed internal files, or scored-label leakage. Excluded third-party material must be explicitly identified with acquisition and licence boundaries. The release manifest, public-text hygiene scan, unit tests, integration tests, and clean-environment validation must all pass.
