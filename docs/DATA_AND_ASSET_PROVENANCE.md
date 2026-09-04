# Data and asset provenance

Version 1.0 follows a user-supplied-data policy. No third-party molecular row table is bundled, including sources whose recorded terms may permit redistribution. This uniform project decision must not be read as a claim that all upstream licences are restrictive. `configs/redistribution_decisions.json` records the licence finding and the project bundling decision separately for every primary and supplementary source.

## Dataset decisions

| Source | Recorded rights finding | Version 1.0 decision |
|---|---|---|
| Llompart AqSolDBc | Etalab Open Licence 2.0 | Not bundled; obtain version 2.0 and run the hash-pinned preparer |
| ComPlat | README declares MIT, but no LICENSE file or separate data-rights statement was found at the pinned revision | Not bundled; supply the pinned Train/Test files locally |
| TDC AqSolDB | TDC software is MIT; dataset rights are separate | Not bundled; supply the two historical runtime exports locally |
| JCheM | Repository is MIT; the included dataset has no separate data notice | Not bundled; supply the pinned workbook locally |
| AqSolDB repository data | Data directory marked CC0-1.0 | Not bundled under the uniform policy; obtain it upstream where needed |
| Biogen pH 6.8 CLND panel | No standalone row-level redistribution grant was recorded | Not bundled; aggregate Supplementary results only |
| Ghanavati panels | No repository licence identified | Not bundled; aggregate Supplementary results only |
| ChemBERTa base model | No explicit licence in the retrieved metadata | Weights not bundled or automatically downloaded |

The ChemBERTa revision observed during the provenance audit is recorded separately from the historical training identity. The historical run retained the model identifier but did not preserve an immutable repository revision, so the audit-observed revision is not presented as the historical revision.

## Derived asset decisions

Per-molecule component and final predictions, language and geometry embeddings, conformer coordinates, graph caches, fitted checkpoints, experimental labels, and fold-labelled molecular tables are excluded. These objects retain direct links to third-party molecular rows, and some also depend on pretrained resources whose redistribution terms are unresolved.

Original DLG-Sol code, documentation, and aggregate or configuration-level result tables are included. The machine-readable Supplementary files distributed with the article contain the public supporting result summaries without molecular structures or row-level experimental labels.

## Reproducibility cost

Without lawfully obtained upstream row-level inputs, a third party cannot independently recompute the article's per-row RMSE differences or paired confidence intervals from this repository alone. The public aggregate tables and hash-bound parity manifests support integrity auditing, while the local-package preparers and scoring workflow support row-level recalculation for users who obtain the required upstream inputs. Hash verification is not a substitute for access to the rows themselves.
