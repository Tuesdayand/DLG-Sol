# DLG-Sol v1.0.1 reproducibility package

This patch synchronizes the public reproducibility package with the revised manuscript and Supplementary Information.

## Corrections and additions

- Corrects Article Table 2 to use the additive Bhattacharya--Roy MLP--GNN as the primary independently retrained comparator.
- Retains the interaction-augmented Bhattacharya--Roy model only as a prespecified sensitivity analysis.
- Adds a locked Article Table 2 contract and regression checks for comparator identity, displayed RMSE values, and machine-readable table membership.
- Replaces Supplementary-table-number filenames with stable content-based filenames.
- Adds 99% confidence intervals for external-comparator and component comparisons and adds the machine-readable main-evaluation effects underlying Article Table 3.
- Adds the MolPROP source-reproduction audit summary.
- Adds a machine-readable environment matrix and explicit environment-isolation guidance.

The release continues to use the documented user-supplied-data policy. It does not redistribute third-party molecular rows, historical checkpoints, embeddings, predictions, or model weights.
