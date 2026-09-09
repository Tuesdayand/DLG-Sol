# Machine-readable Supplementary data

This directory uses content-based file names so that files remain unambiguous if Supplementary tables are renumbered. The current table and figure mappings are listed in the Supplementary Information.

## Interpretation rules

- RMSE is the primary metric and MAE is secondary.
- Negative DLG-Sol-minus-comparator or DLG-Sol-minus-component differences favour DLG-Sol.
- Confidence-interval levels are explicit in column names. The `significance_tier` and `star` fields distinguish effects resolved at 95% (`**`) and 99% (`***`) confidence.
- External-comparator evidence classes are not interchangeable. Released predictions, same-split reproductions, schema-and-setting adaptations, independently retrained adaptations, and prespecified sensitivity analyses answer different questions.
- Chemical-subgroup rows are exploratory paired-row bootstrap results without multiplicity correction.
- The 16 subgroups are levels of seven separately analysed chemical axes, not a Cartesian partition.
- `sparse_n_lt_100` identifies the four evaluation-by-subgroup rows with fewer than 100 molecules.
- MolPROP rows document a source-reproduction audit and are not manuscript performance comparisons.

## Files

- `main_evaluation_effects_12_rows.csv`: the 12 DLG-Sol-versus-component effects underlying Article Table 2, with RMSE and MAE point estimates and 95% and 99% RMSE confidence intervals.
- `source_paper_benchmark_context.csv`: the 18 source-paper context rows underlying Article Table 3 and Supplementary Table S20, including available RMSE and MAE values and evidence qualifications.
- `external_comparator_metrics_31_rows.csv`: external-comparator RMSE, MAE, bias, 95% and 99% paired intervals, and evidence class. The legacy `main_table3_comparison` field marks the 25 main-text contrasts retained across current Article Tables 3--4; its name is preserved for schema stability.
- `descriptor_preprocessing_record.csv`: dataset-specific descriptor filtering and retained-feature counts.
- `single_representation_grid_185_rows.csv`: complete available single-representation grid.
- `pair_representation_grid_315_rows.csv`: complete representation-pair grid.
- `developmental_single_representation_screen_metrics.csv`: complete 35-candidate developmental single-representation screen across the recorded validation and development panels.
- `developmental_pair_representation_screen_metrics.csv`: complete 105-candidate developmental pair-representation screen across the recorded validation and development panels.
- `developmental_focused_tuning_metrics.csv`: focused 12-candidate developmental tuning records for the selected representation pair.
- `chemical_subgroup_delta_rmse_96_rows.csv`: all 96 subgroup RMSE comparisons.
- `chemical_subgroup_delta_mae_96_rows.csv`: all 96 subgroup MAE comparisons.
- `seed_registry.csv`: compact seed summary.
- `molprop_source_reproduction_summary.csv`: source-reproduction audit summary for MolPROP.

Row-level matched predictions and raw benchmark data are not included under the completed dataset-specific redistribution review and documented user-supplied-data policy.
