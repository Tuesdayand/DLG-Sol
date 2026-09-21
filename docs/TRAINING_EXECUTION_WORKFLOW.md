# Training-execution workflow

`configs/training_execution_recipes.json` connects each audited training-role unit to the public descriptor, language, geometry, fusion, and fixed-blend configurations. The recipes reference the row-use contract by evaluation ID, branch ID, and all declared folds. They do not duplicate row counts or record-identity hashes from `configs/training_role_contracts.json`.

The four branch command-line entry points consume one directory produced by `scripts/materialize_training_unit.py`. They do not accept an original benchmark table or a scored-label file. Each command writes label-free `record_id,prediction` output and a hash-bound `run_manifest.json`.

## Supported execution boundary

The intended path is:

1. validate or reconstruct a user-local benchmark package;
2. materialize one evaluation, branch, and fold into stage-specific views;
3. run the corresponding branch training or reuse recipe;
4. write label-free component predictions and required embeddings;
5. assemble fixed-blend predictions and score them with the existing evaluation runner.

The target release does not promise a one-command six-evaluation controller, automatic benchmark acquisition, cluster scheduling, historical checkpoint identity, or equality between fresh-training metrics and the article's recorded metrics. Fresh training is a protocol reproduction.

## Evaluation-specific execution modes

| Evaluation | Descriptor, language, and geometry | Fusion heads | Blend coefficient |
|---|---|---|---|
| R01 | Unit-specific training; descriptor selection is nested within each outer fold | Four fixed-schedule heads | Prespecified alpha of 0.5 |
| R02 | One frozen global ComPlat Train/Val preselection followed by outer-fold refits | Four fixed-schedule heads per outer fold | Cross-fitted from the other four OOF folds |
| R03 | Fold-local train-only selection | Four fixed-schedule heads per outer fold | Cross-fitted from the other four OOF folds |
| R04 | Reuse the corresponding R03 split models on the supplied test | Reuse the corresponding R03 heads | Fit once from complete R03 OOF predictions |
| R05 | Use each reported Train/Val split | Four fixed-schedule heads fitted on Train | Fit within the corresponding reported validation split |
| R09 | Reuse the R02-selected settings for a full ComPlat Train refit | Four fixed-schedule heads fitted on full Train | Fit once from complete R02 OOF predictions |

## Branch commands

Install the branch-specific environment before running a command. The examples below assume that `UNIT_DIR` is an already validated and materialized unit and that each output directory is new or empty.

```bash
PYTHONPATH=src python scripts/train_descriptor_unit.py \
  --materialized "$UNIT_DIR" \
  --output "$RUN_DIR/descriptor" \
  --descriptor-cache "$LOCAL_MORDRED_CACHE"

PYTHONPATH=src python scripts/train_language_unit.py \
  --materialized "$UNIT_DIR" \
  --output "$RUN_DIR/language" \
  --device cuda:0

PYTHONPATH=src python scripts/train_geometry_unit.py \
  --materialized "$UNIT_DIR" \
  --output "$RUN_DIR/geometry" \
  --device cuda:0

PYTHONPATH=src python scripts/train_fusion_unit.py \
  --materialized "$FUSION_UNIT_DIR" \
  --language-run "$RUN_DIR/language" \
  --geometry-run "$RUN_DIR/geometry" \
  --output "$RUN_DIR/fusion" \
  --device cuda:0
```

The descriptor cache is optional. When omitted, the descriptor command calculates Mordred descriptors from the materialized SMILES. A supplied cache must be label-free and keyed by unique `record_id` values.

R04 commands require `--reuse-run` pointing to the corresponding R03 same-fold branch run. R09 language and geometry commands require `--settings-run` pointing to an R02 run carrying the frozen settings. The R09 descriptor settings are reconstructed from the recorded descriptor protocol. Fusion R04 likewise requires the same-fold R03 fusion run.

The ChemBERTa and 3D-MPNN commands derive the recorded dataset- and fold-specific seeds and training profiles from their configuration files. `--seed-offset` creates an explicitly distinct replication without changing the primary profile. The 3D profiles preserve fixed transferred parameters for R01 and R05, the frozen ComPlat settings and 19-epoch refit for R02/R09, and the four-candidate fold-local search used for TDC R03.

## ComPlat global-preselection boundary

The primary R02 workflow is `frozen_global_preselection`. It is not fold-local reselection. The pinned original ComPlat training file is grouped by RDKit 2023.09.6 InChIKey, group-mean logS is stratified with the recorded quantile rule, and `StratifiedShuffleSplit` with random state 260525 reconstructs 14,351 selection-fit rows and 3,586 selection-score rows.

The Gate B2 audit regenerated this split from the pinned 17,937-row source. Both record-ID hashes matched the training-role contract, and membership matched the historical Train/Val metadata with zero discrepancies. The separate fold-local reselection result is a sensitivity analysis rather than the primary R02 workflow; it is reported in the SI section "ComPlat selection sensitivity" and Supplementary Table S6.

Future branch entry points must reject an R02 primary run that requests fold-local reselection unless an explicit non-primary sensitivity mode is implemented and clearly labelled. Every run manifest must record `selection_mode`.

## Output and privacy contract

The common prediction schema is `record_id,prediction`. Scored inputs and branch outputs used before the scoring phase must not contain observed labels. Run manifests must record configuration and materialization hashes without recording private absolute paths, host names, credentials, or internal development labels.

Historical molecular rows, labels, predictions, embeddings, and checkpoints are not redistributed. `results/verified_manifests/branch_execution_integration.json` records a user-local JCheM descriptor run that traversed input preparation, role materialization, loading of the pinned historical raw-descriptor cache, actual XGBoost fitting, validation selection, and label-free prediction. Mordred recomputation from SMILES was not exercised in that integration. This is an execution smoke test, not a claim that all six evaluations were freshly retrained or that fresh weights reproduce archived checkpoints.
