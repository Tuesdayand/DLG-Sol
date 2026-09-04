# ChemBERTa language branch

## Scope

Gate L1 exports three molecular-language encoder contracts evaluated during development:

1. canonical-SMILES multitask fine-tuning;
2. single-task fine-tuning with two randomized training SMILES per molecule;
3. canonical-SMILES single-task fine-tuning.

Only the second variant, single-task fine-tuning with two randomized training SMILES per molecule, supplies the final `aug2_head4` neural component. The multitask and canonical single-task variants are preserved as development candidates and are not final DLG-Sol members. All three initialize from `seyonec/ChemBERTa-zinc-base-v1`, fine-tune every encoder layer, tokenize to 256 positions, and use a masked mean over non-padding final-layer states. Inference always uses the canonical SMILES and produces one 768-dimensional embedding per molecule. The exact settings and their scopes are machine-readable in `configs/chemberta_variants.json`.

The same file records evaluation-specific execution profiles. R01, R03, and R05 select a checkpoint within their declared development partition. R02 uses the frozen ComPlat development selection of 14 epochs for each outer-fold refit, and R09 transfers that epoch count to its full-training refit. R04 reuses the corresponding R03 encoder. `scripts/train_language_unit.py` accepts a materialized language unit and writes label-free predictions, embeddings, model state, selection information, and a run manifest. `--seed-offset` creates an explicitly distinct replication.

The historical records did not preserve the immutable revision identifier of the initial pretrained repository. The identifier is known, and each resulting fine-tuned checkpoint is hash-audited internally, but the release does not claim bitwise reproducibility of the original pretrained download. Fine-tuned checkpoints are not included in this gate.

## Randomized-SMILES variant

`ChemBertaDataset` preserves the historical data layout: one canonical record followed by two randomized records for every fit-partition molecule. RDKit generates a new noncanonical traversal when an augmented record is fetched. Selection and scored inference do not use augmentation.

This augmentation is stochastic at the string level. Reproducing a historical fine-tuned checkpoint therefore requires the archived checkpoint; rerunning from the model identifier and recorded seed is a protocol reproduction rather than a guarantee of bitwise-identical weights.

## Auxiliary multitask development candidate

The non-final multitask candidate jointly predicts logS and five RDKit-derived training targets:

- Crippen logP;
- topological polar surface area;
- molecular weight;
- hydrogen-bond donor count;
- hydrogen-bond acceptor count.

Auxiliary targets are standardized with fit-partition means and sample standard deviations. Their mean-squared error is multiplied by 0.15 and added to the logS loss during fitting. Checkpoint selection uses selection-partition logS RMSE only. The auxiliary output is discarded before fusion; only the pooled 768-dimensional embedding continues to the geometry-fusion branch.

The historical development table contained no invalid molecule and no missing auxiliary value. The public API fails on non-finite fit auxiliary targets instead of silently estimating an all-row fallback.

The internal parity audit recomputes all 10,298 archived auxiliary rows with a `1e-12` absolute tolerance and reloads each of the three archived fine-tuned checkpoints to compare a fixed 16-row embedding probe with a `3e-6` absolute tolerance. The latter tolerance covers CPU-versus-GPU floating-point differences; it is not a training-performance tolerance.

## Public API

Configuration and dependency-light utilities are available from `dlg_sol.language`:

- `load_chemberta_variants`;
- `canonicalize_smiles` and `randomized_smiles`;
- `compute_auxiliary_targets`, `fit_auxiliary_scaler`, and `transform_auxiliary_targets`;
- `masked_mean_pool`.

The PyTorch/Transformers implementation is in `dlg_sol.language.modeling`:

- `ChemBertaDataset`;
- `ChemBertaRegressor`;
- `ChemBertaMultiTaskRegressor`;
- `build_chemberta_model`;
- `extract_embeddings`.

The canonical fine-tuning and label-free inference loop is in `dlg_sol.language.training`:

- `fit_chemberta_variant` fits on the fit partition, applies early stopping using selection-partition logS RMSE, and restores the best state dictionary;
- `predict_chemberta` accepts only scored SMILES and returns predictions plus 768-dimensional embeddings without a scored-label argument.

Install `environment/language-requirements.txt` before importing the modelling module.

For the repository-wide test suite, install `environment/release-test-requirements.txt`; the language-only file intentionally omits XGBoost and Mordred.

## Label firewall

Fit labels and selection labels are development-side inputs. Selection logS RMSE is the sole checkpoint criterion. Scored rows must be supplied to `ChemBertaDataset` without labels or auxiliary targets. Dataset-specific source adapters and complete fold runners remain outside Gate L1; they must use the common fit/selection/scored contract documented in `TRAINING_ORCHESTRATION_API.md`.

## Redistribution boundary

This gate does not redistribute raw SMILES, experimental labels, row-level embeddings, predictions, pretrained files, or fine-tuned checkpoints. The historical-parity manifest contains only hashes, shapes, aggregate differences, and protocol facts.
