# Manuscript and Supplementary Information alignment

The DLG-Sol reproducibility package provides the code and numerical results for **DLG-Sol combines molecular descriptors with language and three-dimensional embeddings for aqueous solubility prediction**. Version 1.0.5 clarifies the documentation without changing the modelling implementation, configurations, or numerical results from v1.0.4.

## Manuscript files checked for v1.0.5

The checked manuscript, SI, and bibliography are identified by filename and SHA-256 hash in `results/verified_manifests/release_v1_0_5_validation.json`. That record also states which release versions the checked files cite. The source filenames retain their original 20 September date; the hashes identify the actual files checked on 22 September 2026. These are specific file checks, not a claim that all future submission files will be identical.

The checked submission uses separate Code availability and Data Availability sections, explicit dataset sources, and contextual references to SI Tables S14–S16. Article Tables 2 and 4 use the dagger notation described below. Table 3 describes its source-paper values as compiled benchmark values and retains the separate qualification for Consensus GNN RMSE recomputed from released predictions. Bibliography wording and manuscript formatting do not change the released numerical results.

## Historical v1.0.4 snapshot

The source filenames retain their original 20 September date. SHA-256 fingerprints in `results/verified_manifests/release_v1_0_4_validation.json` identify only the snapshot checked on 22 September 2026 during v1.0.4 preparation. They do not identify the later submission files or assert that every manuscript statement and table symbol remains unchanged.

## Subsequent submission-format clarification (22 September 2026)

The submission-format clarification prepared after v1.0.4 publication used manuscript and SI files citing v1.0.4. They separated Code availability from Data Availability, provided dataset access links and source citations, and explicitly connected the post-development comparisons to SI Tables S14–S16. The developmental screen remains SI Table S19; SI table numbering is unchanged.

Article Tables 2 and 4 now use `†` instead of `***` for their 99% paired-confidence-interval criteria. Table 2 marks individual DLG-Sol-minus-component intervals entirely below zero. Table 4 marks an evaluation only when all four displayed DLG-Sol-minus-comparator 99% intervals are entirely below zero. Thus its JCheM display is `0.614†`; the comparison against the original authors' released Consensus GNN predictions is still separate. See the table-specific legends: the dagger used for the OCHEM overlap qualification in Article Table 3 has a different, explicitly stated meaning.

These are manuscript presentation changes. The 14 released CSV files, their legacy star codes, numerical values, confidence intervals, model code, and historical release-validation records are unchanged. The release's existing annotation contract and validator continue to check legacy codes; the current manuscript displays their qualifying 99% results with a dagger. This documentation update does not replace or retag v1.0.4.

The journal manuscript and SI are separate submission files, not files compiled by this software repository. The article/SI-to-CSV mapping is in `supplementary/machine_readable/README.md`. All 14 public CSV files match the submission package byte for byte. The 13 CSV files already present in v1.0.3 remain unchanged.

## Checks documented during v1.0.4 preparation

- **Historical Table 4 annotations:** the asterisks in the release-checked snapshot apply to the four external models shown in that column. The original authors' released Consensus GNN predictions belong to the separate Table 3 comparison, not this annotation. That snapshot displayed JCheM as `0.614***` (now `0.614†` in the later manuscript). The integrity validator recomputes the legacy annotation from the existing paired confidence intervals and the model identities in `configs/article_table_4_contract.json`.
- **TDC partitions:** the supplied 7,985/1,997 train/test partition is scaffold-based. The internal five-fold OOF evaluation and validation subsets are logS-stratified, not scaffold splits. See `docs/BENCHMARK_INPUT_API.md` and the unchanged `configs/tdc_partition_recipe.json`.
- **Dataset attribution:** the Methods in the release-checked snapshot directly cited Sorkun et al. for AqSolDB. That attribution is retained in the later submission; citation numbering is maintained in the manuscript, not encoded as fixed numeric references in this repository.

These changes do not alter model fitting, evaluation membership, predictions, aggregate scores, or the scope of the six primary evaluations. The validation record separates software tests and table/figure regeneration from historical training and row-level scoring evidence. No new training or benchmark evaluation was performed for this patch.

## Publication boundary

The v1.0.4 validation record attests to its historical local source snapshot. That snapshot cited v1.0.3; the submission-format files checked after publication cited v1.0.4. The historical `software_link_in_checked_manuscript` field therefore remains `v1.0.3`: it describes that checked file, not a later submission. The v1.0.5 record identifies its own checked files and cited versions separately. New manuscript fingerprints do not retroactively change historical records. The journal submission files remain separate from this repository, and the published v1.0.4 tag is not replaced or moved.
