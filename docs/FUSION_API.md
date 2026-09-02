# Language–geometry fusion API

## Canonical model identity

The final DLG-Sol neural component is `aug2_head4`. It uses one single-task ChemBERTa encoder trained with a canonical SMILES plus two randomized training SMILES per molecule, one benchmark-specific distance-aware 3D representation, and four independently initialized residual regression heads. It is not the heterogeneous mixed four-member development candidate.

`configs/fusion_ensemble.json` is the machine-readable final-model contract. `src/dlg_sol/fusion/` provides configuration validation, fit-only preprocessing, residual-head training, label-free inference, failed-geometry fallback, and four-head averaging.

## Shared representation and four heads

The 768-dimensional ChemBERTa embedding and the fitted 3D embedding with 19 auxiliary molecular features are processed by modality and concatenated. The same resulting vector is supplied to all four heads. Each head has a 384-unit input projection, two residual blocks, dropout 0.45, and a scalar output.

The final head schedule is:

| Head | Base seed | Fixed epochs |
|---:|---:|---:|
| 0 | 260901 | 29 |
| 1 | 260902 | 82 |
| 2 | 260903 | 40 |
| 3 | 260904 | 21 |

Fold-based runs use `base seed + 100000 × fold index`. The fixed epoch counts were the per-seed median best epochs from AqSolDBc five-fold development. This is development provenance, not a claim of prospective selection.

## Preprocessing and dataset adapters

Each modality is standardized separately using the fitted model's training rows, after which the two blocks are concatenated directly. PCA and blockwise L2 normalization are not used by the final model.

The historical dataset adapters applied feature-wise 0.1st–99.9th percentile winsorization using training rows for AqSolDBc, ComPlat, and TDC. The JCheM adapter did not apply winsorization. The core API therefore requires this choice to remain explicit rather than silently enabling it for every dataset.

`configs/dataset_adapters.json` and `docs/DATASET_ADAPTER_API.md` record the evaluation-specific partition roles, fold numbering, coefficient boundary, and scored-row aggregation. In particular, TDC head fitting uses only `partition=fit`, JCheM coefficients use `split=Val`, and the R04/R05 neural predictions are means across five split models.

`fit_fusion_preprocessor` retains the generic PCA/L2 method solely for historical development-candidate and representation-grid reproduction. That route must not be described as part of the final DLG-Sol architecture.

## Failed geometry rows

`fit_training_mean_fallback` computes the mean over finite training-partition geometry embeddings. Its transformer replaces explicitly failed or non-finite geometry rows with that fixed vector. It receives no solubility labels and does not use selection or scored embeddings.

## Training and inference

`fit_fixed_epoch_fusion_head` implements the final head policy: residual MLP, mean-squared error, AdamW, learning rate `5e-4`, weight decay `5e-4`, batch size 128, and gradient clipping at 5. The caller supplies the recorded head-specific seed and epoch count.

`fit_fusion_model`, `PlainMLP`, and the PCA/L2 preprocessor remain available only to reproduce historical candidate-grid experiments. They are not final-model APIs.

`mean_member_predictions` requires a finite matrix with exactly four columns and computes the row-wise float32 arithmetic mean. It has no molecule-specific weights. The separate DLG-Sol fixed blend combines this completed neural prediction with the completed descriptor prediction.

## Verification and redistribution boundary

`results/verified_manifests/fusion_branch_parity.json` records row-level identity checks between the six locked neural predictions and their `aug2_head4` sources without redistributing those rows. `results/verified_manifests/fusion_mixed_candidate_parity.json` preserves the earlier mixed-candidate preprocessing and aggregation audit as development history and explicitly marks it as non-final.

Row-level embeddings, predictions, conformers, and checkpoints are not redistributed in this staging snapshot. The complete final head checkpoint set is not required to establish row-level prediction identity, but a fresh training run is protocol reproduction rather than a claim of bitwise-identical weights.
