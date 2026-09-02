# Dataset adapter API

The final `aug2_head4` architecture is shared across evaluations, but the recorded data partitions are not interchangeable. `configs/dataset_adapters.json` preserves those differences explicitly instead of presenting winsorization, fold numbering, or scored-row aggregation as universal steps.

## Evaluation-specific contracts

| Evaluation | Head-fit rows | Coefficient rows | Scored rows | Fold indices | Winsorization | Scored prediction aggregation |
|---|---|---|---|---|---|---|
| R01 AqSolDBc OOF | `split=Train` | none; alpha is 0.5 | `split=External` | 0–4 | train-only 0.1st–99.9th percentile | one held-out prediction |
| R02 ComPlat OOF | `split=Train` | training-side OOF, outside the head adapter | `split=External` | 0–4 | train-only 0.1st–99.9th percentile | one held-out prediction |
| R03 TDC OOF | `partition=fit` | training-side OOF, outside the head adapter | `partition=oof_holdout` | 0–4 | train-only 0.1st–99.9th percentile | one held-out prediction |
| R04 TDC supplied test | `partition=fit` | complete TDC OOF, outside the head adapter | `partition=official_test` | 0–4 | train-only 0.1st–99.9th percentile | mean of five split-model predictions |
| R05 JCheM supplied test | `split=Train` | `split=Val` within each reported split | `split=External` | 1–5 | none | mean of five split-model predictions |
| R09 ComPlat supplied test | full `split=Train` refit | complete ComPlat OOF, outside the head adapter | `split=External` | full-refit index 0 | train-only 0.1st–99.9th percentile | one full-refit prediction |

TDC `selection` rows are not included in neural-head fitting. They remain reserved in the R03 and R04 adapters. JCheM is the only primary evaluation in which the adapter exposes coefficient-development rows directly, because each reported split fits its blend coefficient from `Val` labels.

## Public functions

`load_adapter_registry` validates the six recorded contracts and the standard output schema. `build_adapter_masks` accepts label-free row metadata, rejects undeclared partition values, and returns disjoint fit, coefficient, scored, and reserved masks. Labels must be supplied separately to the model-fitting or coefficient-fitting APIs.

`adapter_head_schedule` applies the recorded `base seed + 100000 × fold index` rule. Consequently, zero-based R01–R04 folds start at the base seeds, while the one-based JCheM splits start at 360901–360904.

`fit_adapter_preprocessor` selects the dataset-specific train-only winsorization policy and then applies modality-wise standardization and raw concatenation. It cannot enable PCA or L2 normalization for the final model.

`aggregate_scored_predictions` enforces one held-out prediction per OOF record, all five split-model predictions for R04 and R05, or one full-refit prediction for R09. Its input is label-free and its output uses the canonical `record_id` and `aug2_head4` columns.

## Scope

The adapters encode how already permitted, benchmark-specific row tables are mapped into the final model API. They do not download restricted datasets, expose labels, redistribute embeddings or predictions, or imply that all source datasets followed one identical curation procedure.
