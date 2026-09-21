# Benchmark input-package contract

This contract validates user-supplied molecular structures, experimental labels, and fold roles before they enter the planned end-to-end training workflow. It does not download or redistribute benchmark rows, and validation does not establish that a user has permission to use an upstream resource.

## Package layout

Each evaluation uses a separate local directory containing exactly five files:

| File | Required content |
|---|---|
| `package_manifest.json` | Evaluation identity, pinned source revision, retrieval date, acquisition mode, redistribution boundary, canonicalization policy, split origin, and hashes of the other files |
| `molecules.csv` | `record_id,smiles`; one unique, non-empty structure row per record |
| `labels.csv` | `record_id,logS`; one finite experimental label for every molecule identifier |
| `partitions.csv` | `record_id,fold_index,role`; one role per record within each fold |
| `derivation.json` | Hashed source objects, ordered transformations, record-identifier policy, canonicalization policy, and split-assignment origin |

Allowed normalized roles are `fit`, `coefficient`, `scored`, and `reserved`. The mapping from each upstream dataset's native split values to these roles is fixed in `configs/dataset_adapters.json`. The input contract in `configs/benchmark_input_contracts.json` records which normalized roles must occur in every fold.

## Provenance record

Every source object listed in `derivation.json` must include:

- a source identifier present in `configs/data_acquisition.json`;
- a locator that is a URL or non-absolute logical path, never a private local path;
- the pinned upstream revision and ISO retrieval date;
- its SHA-256 digest and byte size.

Transformations must be numbered consecutively from one. Each entry records the operation and the software or script version used. The canonicalization and split-origin statements must match the package manifest exactly.

## Fold rules

- R01, R02, and R03 require each scored record to occur in exactly one held-out fold.
- R04 and R05 require the same supplied-test records in every reported fold because their final predictions average split-model outputs.
- R05 additionally requires a `coefficient` role in each fold; validation labels in that role are used only for that fold's blend coefficient.
- R09 is a single full-refit evaluation and requires each supplied-test record once.
- Every fold must contain all roles declared for that evaluation, every molecule must receive at least one role, and a record cannot have two roles in one fold.

These checks establish package structure and split identity. The downstream training workflow remains responsible for enforcing stage-specific label access. Final fitting and coefficient estimation must exclude the labels scored by the deployed unit. Model-selection provenance is evaluation-specific and is recorded in `configs/training_role_contracts.json`; notably, historical R02 used one non-nested global Train/Val preselection on the same population later scored out of fold.

## Validation

From the repository root, with `src` on `PYTHONPATH`:

```bash
PYTHONPATH=src python -m dlg_sol.data \
  --evaluation-id R01 \
  --package /path/to/local/R01_input_package
```

A valid package prints a JSON object with `"status": "PASS"`. No files are copied, modified, or uploaded.

## Current provenance status

The Llompart et al. AqSolDBc file, ComPlat train/test CSVs, JCheM workbook, and the two historical TDC runtime exports have independently recorded object hashes in `configs/benchmark_input_contracts.json`.

The R01 package can be prepared from the original AqSolDBc CSV downloaded from Dataverse version 2.0:

```bash
python scripts/prepare_aqsoldbc_input.py \
  --source /path/to/AqSolDBc.csv \
  --output /path/to/R01_input_package \
  --retrieval-date YYYY-MM-DD
```

The preparer rejects any source object that does not match the recorded byte size and SHA-256. It preserves the author identifiers, curated SMILES, targets, and row order, then recreates the five outer folds and fold-specific reserved rows from `configs/aqsoldbc_partition_recipe.json`. The resulting 40,235 role assignments reproduce the historical R01 `fit`, `reserved`, and `scored` identities exactly. It does not download or redistribute the source dataset.

This V1-A package keeps all labels in the validated local `labels.csv`. The V1-B training workflow must expose labels through stage-specific views. For protocols with fold-local selection, a molecule scored in one outer fold remains unavailable to that fold's selection, fitting, and coefficient stages even though the same molecule may be a development row in other folds. R02's recorded global-preselection exception is handled separately and must not be mislabeled as nested selection.

The R02 and R09 packages can be prepared from the two pinned files in the released ComPlat repository:

```bash
python scripts/prepare_complat_input.py \
  --train /path/to/final_unique_train.csv \
  --evaluation-id R02 \
  --output /path/to/R02_input_package \
  --retrieval-date YYYY-MM-DD
```

For R09, use `--evaluation-id R09`, add `--test /path/to/final_unique_test.csv`, and select a separate output directory. R02 reconstructs the prespecified five-fold stratified OOF assignment from the released Train rows. R09 assigns all released Train rows to `fit` and all released Test rows to `scored`. Both copy the author-provided `C_ID`, `smiles_canon`, and `LogS` fields without molecular canonicalization.

The R05 package can be prepared from the pinned JCheM workbook:

```bash
python scripts/prepare_jchem_input.py \
  --source /path/to/dataset.xlsx \
  --output /path/to/R05_input_package \
  --retrieval-date YYYY-MM-DD
```

Run this preparer in the separate environment defined by `environment/input-preparation-requirements.txt`. RDKit 2026.03.1 is pinned because three stereochemical canonical-SMILES strings differ from RDKit 2023.09.6 and the later version reproduces the historical R05 metadata exactly. The preparer excludes the same two unparsable source rows, maps the released `Split1`–`Split5` values to `fit`, `coefficient`, and `scored`, and verifies that all five splits use the same 980 test structures. For these scored rows only, it uses the two-decimal `SExp` values from `predictions_test_set`, matching the row-level labels used for the authors' released Consensus GNN comparison. Development labels retain the full precision in `dataset_split`.

The R03 and R04 packages require the exact two-column TDC runtime exports used in the reported experiments.

The supplied TDC ADMET Benchmark Group train/test partition is scaffold-based (7,985 training/validation rows and 1,997 test rows). The five internal OOF folds are target-stratified, not scaffold splits. They use ten logS quantile bins and seed 260722. Within each outer-training partition, 12.5% of the rows are reserved for validation using the same target bins and seed 260722 plus the zero-based fold index. These internal partitions never include the supplied test rows. The executable recipe is `configs/tdc_partition_recipe.json`; the manuscript clarification does not change it.

Prepare the R03 input package with:

```bash
python scripts/prepare_tdc_input.py \
  --train-val /path/to/TDC_AqSol_benchmark_train_val.csv \
  --test /path/to/TDC_AqSol_benchmark_test.csv \
  --evaluation-id R03 \
  --output /path/to/R03_input_package \
  --retrieval-date YYYY-MM-DD
```

Use `--evaluation-id R04` and a separate output directory for the supplied-test package. A newly retrieved standard TDC ADMET-group cache can preserve the same row order while differing from these historical two-column exports at approximately (10^{-9}) target precision, so its file hashes may be rejected; exact reconstruction requires the two recorded runtime-export objects rather than an assumed equivalent current loader result. Run this preparer in the environment defined by `environment/tdc-input-preparation-requirements.txt`. RDKit 2023.09.6 is required because later canonicalization behavior differs for several organometallic structures and RDKit 2026.03.1 does not parse two historical test strings. The preparer reconstructs five target-stratified outer folds using seed 260722, creates a train-only selection subset inside each outer training set, and preserves both unused role boundaries. R03 scores each train/validation row exactly once as an outer-fold holdout and marks the supplied test as reserved. R04 scores the same supplied-test rows in all five folds and marks each fold's outer holdout and selection rows as reserved. The source rows are not redistributed by this repository.
