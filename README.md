# DLG-Sol

Code and machine-readable results supporting **“DLG-Sol: Integrating descriptor, molecular-language, and three-dimensional representations for aqueous-solubility prediction.”**

DLG-Sol combines a Mordred/XGBoost descriptor model with a neural model derived from ChemBERTa and a distance-aware three-dimensional message-passing representation. The final prediction uses one evaluation-level, molecule-independent blend coefficient.

## Repository status

This is a pre-release staging repository. The currently included files cover the verified machine-readable Supplementary summaries, release-integrity checks, the six-evaluation registry, label-firewall checks, fixed blending, article-level scoring metrics, the canonical Mordred/XGBoost descriptor branch, split-safe descriptor training orchestration, the ChemBERTa language-encoder contracts, the explicit-polar-hydrogen distance-aware three-dimensional graph branch, the canonical `aug2_head4` language–geometry branch, evaluation-specific adapter contracts, and a provenance-checked local-package evaluation runner. Row-level benchmark data, historical model checkpoints, automatic data acquisition, and third-party source code are not part of this staging snapshot.

## Canonical core modules

The first code-export gate is under `src/dlg_sol/`. Its evaluation registry, label firewall, fixed-blend functions, regression metrics, and paired bootstrap are documented in `docs/CORE_API.md`.

Run its synthetic contract tests with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The descriptor API and its protocol-specific filtering differences are documented in `docs/DESCRIPTOR_API.md`. Install `environment/descriptor-requirements.txt` to reproduce the recorded Python 3.9 Mordred/XGBoost stack.

The explicit fit/selection/scored partition contract, candidate selection, nested-trial ranking, and prediction aggregation APIs are documented in `docs/TRAINING_ORCHESTRATION_API.md`.

The final randomized-SMILES single-task ChemBERTa encoder and the two non-final development variants are documented in `docs/LANGUAGE_API.md`. Their modelling implementation requires `environment/language-requirements.txt`. Historical fine-tuned checkpoints and row-level embeddings are not redistributed.

The explicit-polar-hydrogen conformer construction and distance-aware three-dimensional message-passing network are documented in `docs/GEOMETRY_API.md`. Install `environment/geometry-requirements.txt` for this branch. Historical conformer caches, coordinates, checkpoints, and row-level embeddings are not redistributed.

The single language–geometry representation, four independently initialized residual heads, dataset-specific adapter winsorization, label-free failed-geometry fallback, fixed-epoch training, and arithmetic-mean neural aggregation are documented in `docs/FUSION_API.md`. Install `environment/fusion-requirements.txt` for this branch. Historical heterogeneous mixed-member experiments are preserved only as non-final development evidence.

The six evaluation-specific partition, fold-numbering, preprocessing, seed, coefficient-boundary, and scored-row aggregation contracts are documented in `docs/DATASET_ADAPTER_API.md`.

The acquisition registry, local-package format, coefficient reconstruction, blend-before-aggregation rule, label firewall, and final scoring commands are documented in `docs/EVALUATION_RUNNER_API.md`. The registry deliberately disables automatic downloads.

## Included data

The directory `supplementary/machine_readable/` contains:

- dataset-specific descriptor preprocessing records;
- external-comparator metrics and paired confidence intervals;
- complete single- and pair-representation grid summaries;
- chemical-subgroup RMSE and MAE analyses;
- key random-seed records.

These files contain aggregate or configuration-level results and do not contain molecular structures or row-level experimental labels.

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

## Regenerate Table 2 data

```bash
python scripts/build_table_2.py
```

The command creates `results/table_2_absolute_rmse.csv` from the public 31-row external-comparator summary. No row-level molecular data are required.

## Data access

Raw benchmark data are not redistributed in this pre-release snapshot. See `data/README.md` and `configs/external_sources.json` for source locations, fixed revisions, and redistribution decisions.

## Reproducibility scope

See `docs/REPRODUCIBILITY_SCOPE.md` for the distinction between the current summary-data snapshot and the planned full training/inference release.

## Citation

Citation metadata are provided in `CITATION.cff`. The article DOI and archival repository DOI will be added when available.

## License

A repository license has not yet been selected by the copyright holders. Until a license is added, this staging snapshot should not be treated as granting reuse or redistribution rights. Third-party materials remain governed by their respective terms; see `THIRD_PARTY_NOTICES.md`.
