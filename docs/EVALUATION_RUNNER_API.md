# Acquisition and evaluation runner API

## Acquisition registry

`configs/data_acquisition.json` records the exact upstream source, pinned revision, licence status, and permitted acquisition mode for every primary evaluation. Network access is denied by default and every resource has `automatic_download: false`. The runner therefore accepts only a user-supplied local package whose provenance matches this registry; it does not download or redistribute benchmark rows, experimental labels, third-party code, or model weights.

## Local package contract

Each local package contains `package_manifest.json` and the CSV files declared by that manifest. The manifest binds the package to one evaluation ID, dataset ID, canonical `aug2_head4` model identity, upstream revision, retrieval date, derivation record, and exact file hashes, byte sizes, row counts, and schemas. Absolute paths, parent traversal, symbolic links, undeclared files, duplicate record-fold pairs, non-finite predictions, and labels in component-prediction files are rejected.

`scored_component_predictions.csv` contains `record_id`, `fold_index`, `descriptor_prediction`, and `aug2_head4`. Evaluations with fitted coefficients additionally require coefficient-side component predictions and labels. `scored_labels.csv` is declared in the package but is not opened or hashed by the assembly phase. Its contents become accessible only in the separate scoring phase.

## Prediction assembly

`assemble_evaluation_predictions` reconstructs the recorded coefficient and aggregation policy for each evaluation:

- R01 uses the prespecified coefficient 0.5.
- R02 and R03 fit one coefficient for each deployment fold from the other training-side out-of-fold rows.
- R04 fits one coefficient from the complete TDC out-of-fold coefficient set.
- R05 fits one coefficient from each split's validation rows.
- R09 fits one coefficient from the complete ComPlat out-of-fold coefficient set.

Cross-fitted coefficient rows may occur in another fold's scored set, but they cannot overlap the scored rows of the same deployment fold. This fold-conditional rule preserves the recorded cross-fitting design without incorrectly imposing global row disjointness.

For supplied-test evaluations, the runner applies the relevant coefficient to each split-model prediction before averaging predictions for a record. This order is necessary for R05 because its five coefficients differ. R04 uses one common coefficient, so averaging before or after the linear blend is mathematically equivalent; the public runner nevertheless uses blend-then-average consistently.

## Scoring

`score_evaluation_predictions` joins labels by exact record identity and reports RMSE, MAE, and paired-bootstrap RMSE differences for DLG-Sol versus each component. The defaults are 10,000 bootstrap replicates with seeds 20260821 and 20260822. Scored labels are never written to the assembled prediction output.

Command-line entry points are available as:

```bash
PYTHONPATH=src python -m dlg_sol.evaluation assemble --evaluation-id R01 --package /path/to/local-package --output /path/to/assembly
PYTHONPATH=src python -m dlg_sol.evaluation score --assembly /path/to/assembly --labels /path/to/local-package/scored_labels.csv --output /path/to/scoring
```

## Scope boundary

This gate provides provenance-checked local-package ingestion, coefficient reconstruction, prediction aggregation, and final scoring. It does not reconstruct raw benchmark datasets, train descriptor or neural models from raw rows, download upstream resources, or grant redistribution rights for excluded data and weights.
