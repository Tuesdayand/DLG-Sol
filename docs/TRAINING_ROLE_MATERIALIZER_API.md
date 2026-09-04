# Training-role materializer API

`dlg_sol.workflow.materialize_training_unit` converts one validated, user-local benchmark package into the stage views recorded in `configs/training_role_contracts.json`. It covers all 126 evaluation, branch, and fold units without redistributing benchmark rows.

The returned stages are `parameter_fit`, `selection_fit`, `selection_score`, `final_refit`, `coefficient_fit`, and `scored_prediction`. Development and coefficient views contain `record_id`, `smiles`, and `logS`. The scored view contains only `record_id` and `smiles`; label-like columns are physically excluded. A stage that is not used by a unit is returned as an empty table with the corresponding safe schema.

Every non-empty view is checked against the audited row count and sorted-record-ID SHA-256. The combined stage assignment is also checked. Undeclared evaluation, branch, or fold identities fail immediately. Zero-overlap units reject any scored-to-training overlap. The 15 R02 representation units retain their documented non-nested global-preselection overlap while keeping each outer fold out of final fitting and coefficient fitting.

## Source-specific resolution

- R01 roles are resolved directly from its reconstructed outer-fold and reserved-validation assignments. Descriptor materialization also returns four nested inner-selection units per outer fold.
- R02 uses the reconstructed outer folds plus the pinned original ComPlat Train file to reproduce the historical InChIKey-grouped `ComPlatSplit1` preselection. This operation requires RDKit 2023.09.6 because later RDKit versions do not parse two historical charge encodings identically.
- R03 and R04 reconstruct the native TDC `fit`, `selection`, `oof_holdout`, and `official_test` roles before mapping them to the requested stage. This prevents a normalized `reserved` role from being treated wholesale as validation.
- R05 maps the released Train, Val, and fixed Test assignments to the audited stage contract. The Val rows legitimately support both component selection and split-specific coefficient fitting.
- R09 reuses the R02 global preselection for representation settings and requires a validated R02 package for the complete-OOF coefficient dependency.

## Command-line use

Install the stage-resolution environment:

```bash
python -m pip install -r environment/training-role-materialization-requirements.txt
```

Materialize a TDC language unit:

```bash
python scripts/materialize_training_unit.py \
  --package /path/to/R03 \
  --evaluation-id R03 \
  --branch-id language \
  --fold-index 0 \
  --output /path/to/local/R03_language_fold0
```

Materialize an R02 representation unit with the pinned user-supplied ComPlat Train file:

```bash
python scripts/materialize_training_unit.py \
  --package /path/to/R02 \
  --evaluation-id R02 \
  --branch-id descriptor \
  --fold-index 0 \
  --complat-train-source /path/to/final_unique_train.csv \
  --output /path/to/local/R02_descriptor_fold0
```

For the R04 blend-coefficient unit, also pass `--r03-dependency-package /path/to/R03`. For the R09 blend-coefficient unit, pass `--r02-dependency-package /path/to/R02`.

The output directory contains one CSV per stage and a `materialization_manifest.json`. R01 descriptor outputs additionally contain nested inner-selection subdirectories. These files contain third-party molecular structures and development labels and are marked `user_local_do_not_redistribute`.

The materializer establishes row routing and label access. It does not calculate descriptors or embeddings, train a model, provide component predictions for coefficient estimation, or promise bitwise reproduction of historical weights.
