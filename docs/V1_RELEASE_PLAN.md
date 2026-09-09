# v1.0 reproducibility-package plan

## Scope

Version 1.0 will be a user-supplied-data reproducibility package. It will provide an executable path from lawfully obtained benchmark inputs through DLG-Sol training, prediction, fixed blending, scoring, and regeneration of public article outputs. It will not bundle raw third-party benchmark rows, third-party source trees, or model weights without clear redistribution permission.

Fresh training under the recorded protocol is a protocol reproduction. Version 1.0 will not claim bitwise reproduction of historical weights when the original pretrained revision, row-level inputs, or checkpoints cannot legally or technically be redistributed.

## Gate V1-A: input provenance and reconstruction

Status: complete, including dataset-specific redistribution decisions and external-comparator file provenance.

- complete the dataset-specific redistribution decision for every primary and supplementary source;
- document lawful acquisition or reconstruction where permitted;
- define canonical local-package schemas for structures, labels, partitions, and record identifiers;
- preserve source revision, retrieval date, original hash, transformation history, and split assignment;
- audit the provenance of every file required by an external-comparator adapter.

## Gate V1-B: branch-level training and evaluation workflow

Status: complete at the documented unit-level protocol-reproduction boundary.

- connect benchmark input validation and canonicalization to descriptor fitting;
- run benchmark-specific ChemBERTa fine-tuning and embedding extraction;
- run conformer construction, 3D-MPNN fitting, and geometry embedding extraction;
- apply dataset-specific preprocessing and failed-geometry fallback;
- fit and aggregate the four residual fusion heads;
- reconstruct the fold-, split-, or evaluation-specific fixed blend coefficient without scored-label leakage;
- provide unit-level branch commands and documented assembly paths covering the recipes for all six primary evaluations.

The command-line workflow must accept only explicit local inputs, must not download unlicensed resources, and must keep scored labels outside preprocessing, fitting, and coefficient selection.

Version 1.0 does not require one command that automatically retrains all six evaluations. It also does not claim that all six fresh-training runs were completed for the release, that fresh weights equal historical checkpoints, or that fresh-training RMSE values must equal the recorded article values. Those are separate claims from providing an executable protocol-level path.

## Gate V1-C: result regeneration

Status: complete for the public aggregate and configuration-level outputs. This status does not claim fresh six-evaluation retraining.

- reproduce RMSE, MAE, and paired-bootstrap comparisons from generated predictions;
- validate Article Table 3 source-paper context, regenerate Article Table 4 and the chemical-subgroup figure, and expose all machine-readable Supplementary tables supported by public inputs;
- record expected numerical tolerances for fresh training separately from exact deterministic transformations;
- execute full and reduced smoke tests in pinned environments.

## Gate V1-D: GitHub release

Status: `v1.0.0` and the manuscript-concordance patch `v1.0.1` were published on GitHub. Version `v1.0.2` synchronizes the restructured Article Tables 3--4, adds the developmental-screen and source-paper-context data, and restores MAE fields in the component-effect table.

- update `CITATION.cff` to the patch version and final release date;
- update README, data-access documentation, release checklist, and reproducibility scope from development to released status;
- make the validator emit `PUBLIC_RELEASE_READY=YES` only when all v1.0 gates pass;
- create and push an annotated patch-release tag without moving the existing `v1.0.0` tag;
- create and publish a GitHub Release for the patch tag;
- retain the GitHub repository and versioned release URL in the manuscript availability statement.

Version 1.0 uses GitHub as its public code and version-release location. A permanent software-archive identifier may be added to `CITATION.cff` and the manuscript if one is assigned. The article DOI may be added after journal publication.

## Definition of ready

Version 1.0 is ready when a user with lawful access to the required inputs can follow the public documentation and execute the supported unit-level branch workflow without private paths, undisclosed internal files, or scored-label leakage. Readiness does not imply one-command production orchestration or a completed fresh run of all six evaluations. Excluded third-party material must be explicitly identified with acquisition and licence boundaries. The release manifest, public-text hygiene scan, unit tests, required descriptor integration test, and clean-environment validation must all pass.

`GITHUB_RELEASE_PACKAGE_READY=YES` means that technical, provenance, result-regeneration, and citation-metadata checks have passed. `PUBLIC_RELEASE_READY=YES` identifies the validated package intended for the version recorded in `CITATION.cff`. Publication of the tag and GitHub Release is checked separately from the file-integrity validator.
