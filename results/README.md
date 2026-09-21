# Verified results

This directory contains compact article-level outputs and manifests connecting public scripts and inputs to article tables and figures. Large checkpoints and third-party row-level data are not stored here.

`verified_manifests/release_v1_0_4_validation.json` records the checks rerun for the manuscript-alignment patch: 139 software tests, agreement of all 14 public CSV files with the manuscript package, preservation of the 13 previously released CSV files, and public table/figure regeneration. The analysis environment was freshly installed; the full software test suite used an existing modelling environment. Earlier row-level parity and training records below remain historical evidence, not newly repeated experiments.

`table_4_absolute_rmse.csv` is rebuilt from the public 31-row external-comparator summary by `scripts/build_table_4.py`. The generator reads `configs/article_table_4_contract.json`, which locks the final comparator variants, labels, panel order, displayed RMSE values, and Article Table 4 membership. Article Table 3 source-paper context and Article Table 2 component effects are supplied separately in `supplementary/machine_readable/source_paper_benchmark_context.csv` and `supplementary/machine_readable/main_evaluation_effects_12_rows.csv`, respectively.

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

`verified_manifests/training_role_contract_parity.json` binds the 126-unit historical training-role contract to source metadata and implementation hashes. It records the non-nested R02 global-preselection exception separately from the zero-overlap final-fitting and coefficient boundaries and does not redistribute row-level inputs.

`verified_manifests/training_role_materialization_parity.json` records successful materialization of all 126 audited units and 20 nested R01 descriptor-selection units from validated local input packages. It confirms that all stage row identities match the contract, every scored view is physically label-free, and the R04/R09 cross-evaluation coefficient dependencies resolve to complete R03/R02 OOF populations.

`verified_manifests/branch_execution_integration.json` records the branch-command contract and one actual user-local JCheM descriptor integration run. The run used the pinned historical raw-descriptor cache, rather than recomputing Mordred values from SMILES, then reproduced the recorded split-1 feature count and selected trial and emitted 980 finite label-free predictions. No molecular rows, predictions, fitted models, or private paths are redistributed in the manifest.

`verified_manifests/training_execution_recipe_parity.json` binds 30 unit-level branch recipes to all 126 audited training-role units without duplicating row counts or hashes. It also records exact reconstruction of the ComPlat 14,351/3,586 global-preselection membership from the pinned original training file under RDKit 2023.09.6.

`verified_manifests/external_adapter_provenance_audit.json` records the identity check of 18 historical files and four repository revisions used to trace the PNNL, Ali XGB-125D, Bhattacharya--Roy, and Consensus GNN comparison evidence. The third-party files and study-specific adapter wrappers are not redistributed.

`verified_manifests/redistribution_decision_audit.json` records the source-by-source and asset-class distribution review. Molecular rows, experimental labels, source-linked predictions, embeddings, conformers, graph caches, and checkpoints remain excluded. Compact aggregate results, configuration records, and original DLG-Sol code remain included.

`verified_manifests/v1c_result_regeneration_audit.json` binds the Article Table 3 source-context audit, Article Table 4 regeneration, public metric and paired-bootstrap results, Figure 2, machine-readable Supplementary tables, and clean-environment analysis artefacts. It does not claim fresh six-evaluation retraining.
