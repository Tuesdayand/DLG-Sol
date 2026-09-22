# DLG-Sol

Code and machine-readable results supporting **“DLG-Sol: Combining descriptor-based and language–3D models for aqueous solubility prediction.”**

DLG-Sol combines a Mordred/XGBoost descriptor model with a neural model derived from ChemBERTa and a distance-aware three-dimensional message-passing representation. The final prediction uses one molecule-independent blend coefficient within each fitted fold, split, or evaluation model.

The study examines how much combining the two models improves prediction over each component, how the gains vary across datasets and molecular subgroups, and which errors remain after blending. Fixed prediction-level blending is the combination method used to investigate these questions, not a newly introduced ensemble method.

## Repository status

This is **v1.0.5** of the DLG-Sol reproducibility package. It clarifies documentation and updates release metadata; the modelling implementation, configurations, and numerical results are unchanged from v1.0.4. The package includes machine-readable Supplementary summaries, release-integrity checks, the six-evaluation registry, label-firewall checks, fixed blending, article-level scoring metrics, the Mordred/XGBoost descriptor model, the ChemBERTa language encoder, the explicit-polar-hydrogen distance-aware 3D-MPNN, the `aug2_head4` language–geometry model, evaluation-specific data contracts, benchmark-input validation, and training and evaluation commands. Row-level benchmark data, historical model checkpoints, automatic data acquisition, and third-party source code are not included.

The [manuscript alignment record](docs/MANUSCRIPT_ALIGNMENT.md) distinguishes the historical v1.0.4 checks from the manuscript/SI files checked during v1.0.5 preparation. File hashes identify those specific source snapshots; they do not certify that every later manuscript statement, citation, or table symbol is unchanged.

The package provides a documented, executable path from lawfully obtained user-supplied benchmark data through DLG-Sol training, prediction, fixed blending, scoring, and regeneration of the reported public outputs. Raw third-party benchmark rows, third-party source trees, and model weights without clear redistribution permission are not bundled. Their provenance, access conditions, and required local-package schemas are documented instead.

The technical package has completed its input, training-workflow, result-regeneration, provenance, and release-metadata gates. Versioned source releases are provided through GitHub. A permanent archival identifier can be added to the citation metadata if one is assigned. The release criteria are documented in `docs/V1_RELEASE_PLAN.md`.

## Environment isolation

Do not install all requirement files into one environment. Several workflows intentionally pin different RDKit, NumPy, pandas, and Python versions. Use the workflow-specific files and machine-readable mapping in `environment/environments.json`; `environment/release-test-requirements.txt` is a public compatibility-test environment, not a claim that all historical models were trained in one environment.

## Canonical core modules

The first code-export gate is under `src/dlg_sol/`. Its evaluation registry, label firewall, fixed-blend functions, regression metrics, and paired bootstrap are documented in `docs/CORE_API.md`.

Run its synthetic contract tests with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The descriptor API and its protocol-specific filtering differences are documented in `docs/DESCRIPTOR_API.md`. Install `environment/descriptor-requirements.txt` to reproduce the recorded Python 3.9 Mordred/XGBoost stack.

The explicit fit/selection/scored partition contract, candidate selection, nested-trial ranking, and prediction aggregation APIs are documented in `docs/TRAINING_ORCHESTRATION_API.md`.

The evaluation-by-branch historical row-use audit and its disclosed R02 global-preselection exception are documented in `docs/TRAINING_ROLE_CONTRACT.md`. `docs/TRAINING_ROLE_MATERIALIZER_API.md` describes how validated local packages are converted into stage-specific labelled development views and physically label-free scored views. `docs/TRAINING_EXECUTION_WORKFLOW.md` maps those views to the audited recipes and the descriptor, language, geometry, and fusion command-line entry points.

The final randomized-SMILES single-task ChemBERTa encoder and the two non-final development variants are documented in `docs/LANGUAGE_API.md`. Their modelling implementation requires `environment/language-requirements.txt`. Historical fine-tuned checkpoints and row-level embeddings are not redistributed.

The explicit-polar-hydrogen conformer construction and distance-aware three-dimensional message-passing network are documented in `docs/GEOMETRY_API.md`. Install `environment/geometry-requirements.txt` for this branch. Historical conformer caches, coordinates, checkpoints, and row-level embeddings are not redistributed.

The single language–geometry representation, four independently initialized residual heads, dataset-specific adapter winsorization, label-free failed-geometry fallback, fixed-epoch training, and arithmetic-mean neural aggregation are documented in `docs/FUSION_API.md`. Install `environment/fusion-requirements.txt` for this branch. Historical heterogeneous mixed-member experiments are preserved only as non-final development evidence.

The six evaluation-specific partition, fold-numbering, preprocessing, seed, coefficient-boundary, and scored-row aggregation contracts are documented in `docs/DATASET_ADAPTER_API.md`.

The acquisition registry, local-package format, coefficient reconstruction, blend-before-aggregation rule, label firewall, and final scoring commands are documented in `docs/EVALUATION_RUNNER_API.md`. The registry deliberately disables automatic downloads.

The user-supplied structure, label, fold-role, source-object, and transformation-history contract is documented in `docs/BENCHMARK_INPUT_API.md`. This validator checks all six primary evaluation layouts without copying or downloading benchmark rows.

The R01 AqSolDBc input package can be generated locally from the exact author-released CSV with `scripts/prepare_aqsoldbc_input.py`. ComPlat R02/R09, TDC R03/R04, and JCheM R05 packages can likewise be reconstructed from hash-pinned inputs with `scripts/prepare_complat_input.py`, `scripts/prepare_tdc_input.py`, and `scripts/prepare_jchem_input.py`. The recipes reproduce the historical record and fold-role identities while keeping source datasets outside this repository.

## Included data

The directory `supplementary/machine_readable/` contains:

- dataset-specific descriptor preprocessing records;
- external-comparator metrics and paired confidence intervals;
- DLG-Sol-versus-component effects with 95% and 99% confidence intervals;
- complete single- and pair-representation grid summaries;
- chemical-subgroup RMSE and MAE analyses;
- fixed-weight versus molecule-specific blending results for the six primary evaluations and the supplementary Biogen panel;
- key random-seed records;
- the MolPROP source-reproduction audit summary.

These files contain aggregate or configuration-level results and do not contain molecular structures or row-level experimental labels.

The 14 CSV files and their article/SI mappings are listed in `supplementary/machine_readable/README.md`. The weighting results have the opposite difference convention to the component comparisons: `delta_rmse` is molecule-specific gate minus fixed blend, so negative values favour the gate. Biogen remains supplementary; it is not a seventh primary evaluation. No additional benchmark evaluations are introduced in v1.0.5.

## Validate the snapshot

Python 3.9 or newer is sufficient for the integrity check:

```bash
python scripts/validate_release.py
```

The command verifies hashes, file sizes, CSV schemas and row counts, and scans public text files for private paths, credentials, and non-public workflow identifiers.

Figure and table regeneration require Python 3.11 or newer and the pinned analysis environment below.

## Regenerate the chemical-subgroup figure

Install the small analysis environment and render the figure:

```bash
python -m pip install -r environment/analysis-requirements.txt
python scripts/render_chemical_subgroup_figure.py
```

The generator writes PDF, PNG, and editable PPTX files to `figures/generated/`.

## Regenerate Article Table 4 data

```bash
python scripts/build_table_4.py
```

The command creates `results/table_4_absolute_rmse.csv` from the public 31-row external-comparator summary. No row-level molecular data are required. Article Table 3 source-paper context is supplied in `supplementary/machine_readable/source_paper_benchmark_context.csv`, and Article Table 2 component effects are supplied in `supplementary/machine_readable/main_evaluation_effects_12_rows.csv`.

## Data access

Raw benchmark data are not redistributed in this package. See `data/README.md`, `docs/DATA_AND_ASSET_PROVENANCE.md`, and `configs/redistribution_decisions.json` for source locations, rights findings, and project bundling decisions. File-level provenance and interpretation boundaries for the external comparators are documented in `docs/EXTERNAL_COMPARATOR_PROVENANCE.md`.

## Reproducibility scope

See `docs/REPRODUCIBILITY_SCOPE.md` for the exact reproducibility claims and the boundary between public aggregate results and user-supplied row-level inputs.

## Citation

Citation metadata for version 1.0.5 are provided in `CITATION.cff`. A software archive identifier and the article DOI may be added when assigned. Changes from v1.0.4 are documented in `docs/RELEASE_NOTES_v1.0.5.md`; the historical v1.0.4 notes are retained separately.

## License

Original DLG-Sol source code and documentation in this repository are released under the MIT License; see `LICENSE`. Third-party data, software, pretrained models, and other external materials remain governed by their respective terms and are not relicensed by this repository; see `THIRD_PARTY_NOTICES.md`.
