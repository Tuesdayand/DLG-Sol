# DLG-Sol v1.0.4 reproducibility package

Version 1.0.4 packages the code and numerical results supporting "DLG-Sol combines molecular descriptors with language and three-dimensional embeddings for aqueous solubility prediction." Its manuscript checks refer to the snapshot recorded during release preparation, not to every subsequent submission revision.

Documentation clarification (22 September 2026): the changes and validation below describe v1.0.4 preparation. Later manuscript edits are distinguished in [the alignment record](MANUSCRIPT_ALIGNMENT.md). This clarification does not alter the published v1.0.4 tag or its historical validation record.

## Changes

- Updates the title and research description in README and citation metadata.
- Adds the 14-row molecule-specific weighting summary underlying Supplementary Table S18, with explicit gate-minus-fixed RMSE differences and the supplementary status of Biogen.
- Preserves the 13 previously released CSV files without numerical or byte-level changes.
- Widens Figure 2 label spacing and regenerates its PDF, PNG, and editable PPTX from the same 96-row input.
- Updates the displayed released Consensus GNN RMSE in the Article Table 3 validation contract to 0.6571; the original source-paper report of 0.657 remains documented.
- Corrects SI references and provides an article/SI-to-data mapping.
- Records fingerprints of the manuscript/SI snapshot checked on 22 September during v1.0.4 preparation, including the original AqSolDB attribution in Methods.
- Makes the Table 4 asterisk scope explicit in that snapshot: the four displayed external models, not the separate released Consensus GNN prediction comparison. Adds interval-based annotation checks; the snapshot displayed JCheM as `0.614***`. The later submission uses `0.614†` with the same 99% confidence-interval criterion.
- Distinguishes the supplied TDC scaffold train/test partition from the internal target-stratified OOF and validation partitions. The executable partition recipe is unchanged.
- Updates integrity records and adds regression checks for the weighting supplement and current release metadata.

## Unchanged scope

The DLG-Sol training code, hyperparameters, six primary evaluations, and previously released result values are unchanged. This patch adds no new benchmark evaluations and makes no claim of fresh six-evaluation retraining. Raw molecular rows, experimental labels, historical predictions, embeddings, checkpoints, and third-party source code remain excluded under the user-supplied-data policy.

## Validation performed for v1.0.4

The release-preparation checks and their execution environments are recorded in `results/verified_manifests/release_v1_0_4_validation.json`. Historical row-level parity records are distinguished from tests and public-output regeneration rerun for v1.0.4. That record identifies the checked manuscript/SI files by hash; it is not a validation record for later submission edits. Earlier releases and tags are retained.

During v1.0.4 preparation, all 146 software tests passed. Public tables and Figure 2 were regenerated and checked against an independent repository copy. The isolated analysis environment installed during the initial v1.0.4 preparation was reused for that verification. Figure 2 retains the checked manuscript's 192 heatmap cell coordinates, colours, and labels; font rendering can differ between plotting environments.
