# Machine-readable Supplementary data

This directory uses the final Supplementary Table numbering.

## Interpretation rules

- RMSE is the primary metric and MAE is secondary.
- Negative `delta_rmse_dlg_minus_comparator` values favour DLG-Sol.
- External-comparator evidence classes are not interchangeable. Released predictions, same-split reproductions, schema-and-setting adaptations, independently retrained adaptations, and pre-specified controls answer different questions.
- Chemical-subgroup rows are exploratory paired-row bootstrap results without multiplicity correction.
- The 16 subgroups are levels of seven separately analysed chemical axes, not a Cartesian partition.
- `sparse_n_lt_100` identifies the four evaluation-by-subgroup rows with fewer than 100 molecules.

## Files

- `supplementary_note_s03_descriptor_preprocessing_record.csv`: dataset-specific descriptor filtering and retained-feature counts.
- `table_s06_external_comparator_metrics_and_paired_ci_31_rows.csv`: external-comparator RMSE, MAE, bias, paired intervals, and evidence class.
- `table_s10_single_representation_grid_all_185_rows.csv`: complete available single-representation grid.
- `table_s11_pair_representation_grid_all_315_rows.csv`: complete representation-pair grid.
- `table_s13_chemical_subgroup_delta_rmse_all_96_rows.csv`: all 96 subgroup RMSE comparisons.
- `table_s13_chemical_subgroup_delta_mae_all_96_rows.csv`: all 96 subgroup MAE comparisons.
- `table_s18_key_seed_record.csv`: compact seed summary.

Row-level matched predictions and raw benchmark data are not included in this snapshot pending dataset-specific redistribution review.
