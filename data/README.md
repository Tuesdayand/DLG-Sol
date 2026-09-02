# Benchmark data access

Raw molecular rows and experimental labels are not included in this pre-release snapshot. The machine-readable Supplementary files contain only aggregate or configuration-level results.

The study evaluated four primary benchmark sources:

| Study name | Role | Upstream access | Redistribution status in this repository |
|---|---|---|---|
| AqSolDBc | Development out-of-fold evaluation | Derived from AqSolDB; https://github.com/mcsorkun/AqSolDB | Not bundled pending final derivative-lineage package |
| ComPlat | Out-of-fold and supplied-test evaluations | https://github.com/ComPlat/water-solubility-prediction | Not bundled; no repository licence identified |
| TDC | Out-of-fold and supplied-test evaluations | https://tdcommons.ai/ | Not bundled; software licence alone does not establish dataset redistribution rights |
| JCheM | Supplied-test evaluation | https://github.com/nadinulrich/log_Sw_prediction | Not bundled pending dataset-specific review |

Supplementary Biogen and Ghanavati panels are also not redistributed. Dataset retrieval, canonicalization, split construction, duplicate handling, and endpoint qualifications are described in the article and Supplementary Information.

Future row-level release files, if permitted, will include source URLs, source versions, retrieval dates, original hashes, transformation records, split assignments, and row-level identifiers sufficient to audit leakage and overlap.

`configs/data_acquisition.json` records the pinned source and permitted acquisition mode for each primary evaluation. All automatic downloads are disabled. Users who already have lawful access may construct a local, non-redistributed package following `docs/EVALUATION_RUNNER_API.md`; the runner validates provenance, schema, hashes, partition-specific coefficient fitting, prediction aggregation, and label access boundaries.
