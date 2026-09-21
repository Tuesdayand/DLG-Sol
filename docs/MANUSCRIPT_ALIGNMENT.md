# Manuscript and Supplementary Information alignment

This package corresponds to the 22 September 2026 revision of **DLG-Sol combines molecular descriptors with language and three-dimensional embeddings for aqueous solubility prediction**. The source filenames retain their original 20 September date; SHA-256 fingerprints in `results/verified_manifests/release_v1_0_4_validation.json` identify the exact revised main-text and SI sources and PDFs.

The journal manuscript and SI are separate submission files, not files compiled by this software repository. The article/SI-to-CSV mapping is in `supplementary/machine_readable/README.md`. All 14 public CSV files match the revised submission package byte for byte. The 13 CSV files already present in v1.0.3 remain unchanged.

## Clarifications reflected in this package

- **Table 4 annotations:** the asterisks on each DLG-Sol RMSE apply to the four external models shown in that column. The original authors' released Consensus GNN predictions belong to the separate Table 3 comparison, not this annotation. JCheM remains `0.614***`. The integrity validator recomputes the annotation from the existing paired confidence intervals and the model identities in `configs/article_table_4_contract.json`.
- **TDC partitions:** the supplied 7,985/1,997 train/test partition is scaffold-based. The internal five-fold OOF evaluation and validation subsets are logS-stratified, not scaffold splits. See `docs/BENCHMARK_INPUT_API.md` and the unchanged `configs/tdc_partition_recipe.json`.
- **Dataset attribution:** the revised Methods directly cite Sorkun et al. for AqSolDB. Citation numbering is maintained in the manuscript, not encoded as fixed numeric references in this repository.

These changes do not alter model fitting, evaluation membership, predictions, aggregate scores, or the scope of the six primary evaluations. The validation record separates software tests and table/figure regeneration from historical training and row-level scoring evidence. No new training or benchmark evaluation was performed for this patch.

## Publication boundary

The validation record attests to this local source snapshot, not to the existence of a remote tag or GitHub Release. The checked manuscript snapshot still cites the published v1.0.3 release. Its versioned software citation can be updated after v1.0.4 is published; any such edit creates a new manuscript fingerprint. This package does not silently change the submission files.
