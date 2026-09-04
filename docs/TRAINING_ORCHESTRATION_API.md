# Training orchestration API

The training orchestration Gate connects the evaluation registry and label firewall to the descriptor branch without redistributing benchmark rows.

`dlg_sol.training.validate_training_partitions` requires non-empty, unique, pairwise-disjoint fit, selection, and scored record identifiers. Fit and selection rows must carry finite development labels. Scored rows must be label-free.

`dlg_sol.training.prepare_descriptor_unit` fits the descriptor schema on fit rows only, applies the retained column sequence to selection and scored rows, and returns aligned numerical matrices. Descriptor tables must not contain targets.

`dlg_sol.training.evaluate_validation_candidates` trains each supplied candidate on the fit partition, ranks candidates using selection RMSE, and predicts label-free scored rows with the selected model. Optional early stopping uses only the selection partition.

`dlg_sol.training.select_nested_trial` reproduces the AqSolDBc ranking contract: pooled inner-validation RMSE, mean fold RMSE, then trial index. Its final tree count is the median of `best_iteration + 1` across the selected trial's inner folds.

`dlg_sol.training.validate_oof_predictions` requires one finite prediction per expected OOF record. `dlg_sol.training.aggregate_member_predictions` requires the same complete member set for every scored record and returns the arithmetic mean in first-seen record order.

This Gate does not acquire benchmark data, construct dataset-specific split files, or expose row-level predictions. Callers must obtain permitted source data and convert each recorded split into the explicit fit, selection, and label-free scored tables required by the API.

This API is a strict disjoint-role primitive. It does not by itself reproduce historical stage transitions in which a validation row later enters a final refit, nor the disclosed R02 global-preselection exception. Those evaluation-by-branch rules are recorded in `configs/training_role_contracts.json`. Their label-safe implementation is documented in `TRAINING_ROLE_MATERIALIZER_API.md`.
