# Historical training-role contract

`configs/training_role_contracts.json` records which molecular rows were used at each stage of the reported descriptor, language, geometry, four-head fusion, and blend-coefficient workflows. It complements the evaluation-level roles in each benchmark input package. The label-safe implementation described in `TRAINING_ROLE_MATERIALIZER_API.md` materializes these views but does not launch model training.

## Why a second role layer is necessary

The normalized input-package roles (`fit`, `coefficient`, `scored`, and `reserved`) protect evaluation identity, but `reserved` does not have one training meaning across all evaluations. For example, TDC packages use it for both a train-side selection subset and rows belonging to the other reported evaluation. Treating every reserved row as validation data would therefore expose labels that the deployed model must not use.

The historical contract distinguishes six stages:

| Stage | Meaning |
|---|---|
| `parameter_fit` | Rows used to update a deployed model when no later refit followed |
| `selection_fit` | Rows used to fit candidates or establish a fixed schedule |
| `selection_score` | Rows whose labels selected a candidate, checkpoint, or epoch |
| `final_refit` | Rows used after selection was frozen to fit the deployed model |
| `coefficient_fit` | Rows whose component predictions and labels fixed the blend coefficient |
| `scored_prediction` | Rows receiving the reported prediction; their labels must be absent from training views |

Reuse across stages is not automatically leakage. A validation row may enter a final refit after selection has finished if that row is not scored by that deployed unit. In R05, the same reported validation rows legitimately support component selection and split-specific coefficient fitting. The contract instead checks whether a row's label enters the selection, deployed fitting, or coefficient stages that produce its own reported prediction.

## R02 historical selection boundary

R02 is ComPlat OOF after one global Train/Val preselection. Descriptor hyperparameters and language and geometry training epochs were selected on a 14,351/3,586 split of the same 17,937-row population later scored out of fold. Final component fitting, four-head fitting, and fold-specific coefficient fitting excluded each held fold, but the upstream preselection was not nested within the outer folds.

This exception is explicit in the contract and in `configs/evaluations.json`. It is not counted as a zero-overlap label firewall. A fold-local reselection sensitivity analysis retained the main comparison conclusion, while the reported primary estimates remain those from the recorded historical workflow.

## Cross-evaluation coefficient sources

R04 obtains one coefficient from the complete R03 OOF component predictions. R09 similarly obtains one coefficient from the complete R02 OOF predictions. These dependencies belong to coefficient orchestration and must not be inferred by promoting all reserved rows in an R04 or R09 package to coefficient rows.

## Audit evidence and implementation status

`results/verified_manifests/training_role_contract_parity.json` binds the contract to the historical metadata and source-script hashes. It summarizes 126 evaluation-by-branch units. No molecular rows, labels, embeddings, predictions, or checkpoints are included.

The status `AUDITED_MATERIALIZER_AVAILABLE` indicates that the 126 audited units can be reconstructed from lawfully obtained, validated local benchmark packages. `results/verified_manifests/training_role_materialization_parity.json` records full contract parity, including all 20 nested R01 descriptor units and the 15 disclosed R02 selection exceptions. The existing generic training-orchestration API remains a strict disjoint-role primitive. The materializer supplies the more detailed historical stage transitions without claiming complete end-to-end retraining.
