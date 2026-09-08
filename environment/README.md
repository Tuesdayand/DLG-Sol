# Environments

Create separate environments for the workflows below. Do not install every requirements file into one environment: the pinned RDKit, NumPy, and pandas versions intentionally differ. `environments.json` provides the machine-readable Python and requirements-file mapping. A `recommended_python` entry identifies the public execution target when the exact historical Python version was not recorded; it is not a claim about the original training interpreter.

The integrity validator uses only the Python standard library and supports Python 3.9 or newer.

The canonical core tests use NumPy and pandas from `analysis-requirements.txt` and require Python 3.11 or newer.

Source-specific benchmark packages use `input-preparation-requirements.txt`. Its RDKit version is intentionally newer than the model-training environments because it reproduces the historical JCheM canonical-SMILES metadata. TDC R03/R04 instead use `tdc-input-preparation-requirements.txt`, which pins RDKit 2023.09.6 because that version reproduces the historical organometallic canonicalization and parses every recorded TDC row. The generated strings can then be consumed by the downstream stage-specific environments.

Training-role resolution uses `training-role-materialization-requirements.txt`. It pins RDKit 2023.09.6 so the historical ComPlat InChIKey-grouped Train/Val preselection can be reconstructed from the original, user-supplied SMILES. The resulting stage files remain local and are not redistributable through this repository.

The descriptor branch uses `descriptor-requirements.txt`, which records the Python 3.9 modelling stack used for raw Mordred calculation and the original descriptor-model workflows. The repaired AqSolDBc nested calculation retained the same public filtering and candidate-generation contract but ran its cached descriptors with NumPy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, and XGBoost 3.2.0.

The ChemBERTa language branch uses `language-requirements.txt`. Its pinned stack supports configuration validation, RDKit auxiliary targets and SMILES augmentation, fine-tuning, checkpoint loading, and Transformers embedding extraction. The pretrained repository revision used by the historical run was not recorded, so this environment pin does not by itself imply a bitwise-identical fresh pretrained download.

The 3D-MPNN branch uses `geometry-requirements.txt`, and the four-head residual fusion branch uses `fusion-requirements.txt`. `release-test-requirements.txt` combines the branch environments for full local workflow testing.

The distance-aware 3D graph branch uses `geometry-requirements.txt`. It pins the recorded RDKit graph-construction stack with PyTorch 2.7.1 and PyTorch Geometric 2.6.1 for checkpoint loading, graph reconstruction, and label-free inference.

The language–geometry fusion branch uses `fusion-requirements.txt`. It pins the NumPy, scikit-learn, and PyTorch versions used for preprocessing parity, MLP checkpoint compatibility, training, and inference. Language and geometry embedding generation retain their separate environment files.

Use `release-test-requirements.txt` to run the complete cross-branch test suite without optional-dependency skips. It combines the descriptor, language, and geometry requirement files; it is a validation environment rather than a claim that every reported workflow was trained in one environment.

The v1.0 reproducibility package provides separate locked or stage-specific environments for:

- dataset-specific DLG-Sol orchestration workflows;
- TensorFlow/DeepChem Consensus-GNN adaptation;
- analysis and figure/table regeneration.

These environments are kept separate because the reported workflows used different framework and Python versions. A single unconstrained environment would not faithfully describe the computations.
