# Descriptor branch

The canonical descriptor branch is available from `dlg_sol.descriptors`. It covers raw two-dimensional Mordred calculation, partition-local schema fitting, deterministic protocol-specific XGBoost candidate grids, feature-matrix transformation, and XGBoost construction.

## Filtering contract

`fit_descriptor_schema` applies four ordered stages to the supplied fitting rows only:

1. remove every descriptor containing a missing value;
2. remove constant descriptors;
3. remove descriptors at or below the protocol-specific variance threshold;
4. where enabled, remove highly correlated descriptors.

The input column order is authoritative in stage 4. For a correlated pair, the earlier input descriptor is retained and the later descriptor is removed. AqSolDBc uses variance `ddof=0` and absolute Pearson correlation `>0.95`. TDC and JCheM use variance `ddof=1` and absolute Pearson correlation `>=0.98`. The recorded primary ComPlat protocol does not apply pairwise-correlation filtering.

`transform_descriptor_frame` applies a fitted schema without refitting it and rejects missing retained columns or non-finite output values.

## Candidate grids

`generate_candidates` returns the candidate grid for `aqsoldbc`, `tdc`, `complat`, or `jchem`. TDC and JCheM require a fold or split number. The AqSolDBc grid is stored explicitly because regenerating log-uniform values with different NumPy versions can change the last floating-point bit; the stored values are those used by the recorded NumPy 2.4.4 repair calculation.

## XGBoost construction

`build_xgb_regressor` fixes the squared-error objective, histogram tree method, random seed, device, thread count, and quiet output. `fit_xgb_regressor` optionally binds one explicit validation partition for early stopping. Dataset splitting, candidate scoring, tree-count selection, fold aggregation, and scored-label access remain orchestration responsibilities and must follow `configs/evaluations.json` and `configs/descriptor_protocols.json`.

## Verification

`results/verified_manifests/descriptor_branch_parity.json` records full-row feature-schema parity, candidate-grid hashes, selected trials, source hashes, and tests under both the recorded Python 3.9 modelling stack and the later AqSolDBc repair stack. Row-level molecular data are not redistributed in this gate.
