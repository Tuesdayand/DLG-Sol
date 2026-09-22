# DLG-Sol v1.0.5 — documentation clarification

Version 1.0.5 clarifies the documentation accompanying the DLG-Sol reproducibility package. The modelling implementation, configurations, and all 14 machine-readable result CSVs are unchanged from v1.0.4.

## Changes

- Distinguishes the manuscript/SI files checked during v1.0.4 preparation from the submission files checked for this release. Each validation record identifies a specific snapshot, not every later manuscript version.
- Removes unqualified claims of alignment with a final manuscript and uses neutral descriptions of the study and submission package.
- Explains how the manuscript's dagger annotations correspond to the unchanged legacy star codes in the CSVs, with separate meanings for each table.
- Preserves the historical v1.0.3 citation recorded during v1.0.4 preparation and states the cited versions of the newly checked files separately.
- Updates citation metadata and the release validator's expected version to 1.0.5, and adds checks for this documentation release.

## Unchanged scientific content

No model fitting, hyperparameter search, new benchmark evaluation, or row-level re-scoring was performed for this release. Model code, configurations, figure assets, aggregate scores, confidence intervals, and the 14 public CSV files remain byte-identical to v1.0.4. Historical validation records are retained unchanged. The versioned package still requires lawfully obtained user-supplied molecular data; no new raw data, model weights, or third-party source code are bundled.

## Validation and version history

The checks performed for this release, the checked manuscript file hashes, and the software test results are recorded in `results/verified_manifests/release_v1_0_5_validation.json`. The release manifest covers the files in this package. Earlier validation records and the v1.0.4 tag retain their historical meaning and are not overwritten by this release.
