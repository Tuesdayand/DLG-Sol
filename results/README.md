# Verified results

This directory contains compact article-level outputs and manifests connecting public scripts and inputs to article tables and figures. Large checkpoints and third-party row-level data are not stored here.

`table_2_absolute_rmse.csv` is rebuilt from the public 31-row external-comparator summary by `scripts/build_table_2.py`.

`verified_manifests/core_module_parity.json` records a six-evaluation full-row parity check for the canonical metrics, paired bootstrap, and fixed-blend application without redistributing the row-level inputs.

`verified_manifests/core_clean_environment_smoke.json` records the isolated pinned-environment execution of the nine synthetic core contract tests and the release validator.

`verified_manifests/geometry_branch_parity.json` records exact graph reconstruction on a fixed probe, strict loading of representative archived checkpoints, and development-embedding parity without redistributing molecular rows, conformers, coordinates, checkpoints, or embeddings.

`verified_manifests/geometry_clean_environment_smoke.json` records the pinned cross-branch test execution after the geometry gate was added.

`verified_manifests/fusion_branch_parity.json` records six row-level identity checks between the locked neural predictions and their actual `aug2_head4` sources, the four residual-head contract, and the final preprocessing boundaries without redistributing row-level data.

`verified_manifests/fusion_mixed_candidate_parity.json` preserves the earlier raw/PCA mixed-candidate audit as superseded development evidence. It is not evidence for the final DLG-Sol architecture.

`verified_manifests/fusion_clean_environment_smoke.json` records the pinned cross-branch test execution and validator counterexamples after the fusion gate was added.

`verified_manifests/dataset_adapter_parity.json` records role-mask, preprocessing, seed-schedule, and scored-prediction aggregation parity across 26 historical units spanning all six primary evaluations, without redistributing row-level inputs, embeddings, or predictions.

`verified_manifests/dataset_adapter_clean_environment_smoke.json` records the fresh pinned-environment execution of 65 cross-branch tests and three adapter-validator counterexamples.

`verified_manifests/evaluation_runner_parity.json` records six-evaluation coefficient reconstruction, row-level prediction assembly, metric, and paired-bootstrap parity from provenance-checked local packages without redistributing their contents.

`verified_manifests/evaluation_runner_clean_environment_smoke.json` records the isolated pinned-environment execution of the complete cross-branch and evaluation-runner test suite plus release-validator counterexamples.
