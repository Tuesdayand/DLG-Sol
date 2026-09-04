# Benchmark data access

Raw molecular rows and experimental labels are not included in the v1.0 package. The machine-readable Supplementary files contain only aggregate or configuration-level results.

The study evaluated four primary benchmark sources:

| Study name | Role | Upstream access | Redistribution status in this repository |
|---|---|---|---|
| AqSolDBc | Development out-of-fold evaluation | Llompart et al. replication dataset, https://doi.org/10.57745/CZVZIA | Not bundled; version 2.0 is licensed under Etalab Open Licence 2.0 and the exact upstream object is hash-pinned |
| ComPlat | Out-of-fold and supplied-test evaluations | https://github.com/ComPlat/water-solubility-prediction | Not bundled; the pinned README declares MIT, but no LICENSE file or separate upstream-data rights statement was found |
| TDC | Out-of-fold and supplied-test evaluations | https://tdcommons.ai/ | Not bundled; software licence alone does not establish dataset redistribution rights |
| JCheM | Supplied-test evaluation | https://github.com/nadinulrich/log_Sw_prediction | Not bundled; the MIT repository includes the dataset but provides no separate data notice |

Supplementary Biogen and Ghanavati panels are also not redistributed. Dataset retrieval, canonicalization, split construction, duplicate handling, and endpoint qualifications are described in the article and Supplementary Information. Exact local reconstruction recipes are provided for AqSolDBc R01, ComPlat R02/R09, TDC R03/R04, and JCheM R05. The corresponding scripts under `scripts/prepare_*_input.py` verify source hashes and create local packages without copying source rows into this repository. The TDC recipe is bound to the two historical runtime exports used in the reported experiments because dataset rows are not redistributed and current software availability alone does not establish identical dataset content or numeric precision.

The source-by-source v1.0 decisions are finalised in `configs/redistribution_decisions.json` and explained in `docs/DATA_AND_ASSET_PROVENANCE.md`. Etalab- and CC0-licensed sources remain outside the bundle under a uniform user-supplied-data policy; this project decision does not imply that those licences are restrictive.

`configs/data_acquisition.json` records the pinned source and permitted acquisition mode for each primary evaluation. All automatic downloads are disabled. Users who already have lawful access may construct a local, non-redistributed training-input package following `docs/BENCHMARK_INPUT_API.md`. Prediction assembly and scoring use the separate package described in `docs/EVALUATION_RUNNER_API.md`.
