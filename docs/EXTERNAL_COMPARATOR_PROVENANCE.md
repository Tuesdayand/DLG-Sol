# External-comparator provenance

The article distinguishes model identity from evidence strength. An authors' released prediction, a same-split reproduction, a schema-and-setting adaptation, an independently retrained architecture adaptation, and a prespecified control are not treated as interchangeable replications.

`configs/external_adapter_provenance.json` records the exact upstream revisions, source-file identities, study-local wrapper identities, execution environments, and permitted interpretations for the four external comparator families. The underlying third-party source trees, study-local historical wrappers, contracts, checkpoints, and row-level predictions are not bundled.

## Evidence classes

| Comparator | Evaluations | Evidence class | Interpretation boundary |
|---|---|---|---|
| PNNL GNN | R01, R02, R03, R04, R05, R09 | Independently retrained adaptation | Split-matched architecture comparison; the original study population was not identifiable |
| Ali XGB-125D | R01, R03, R04, R05 | Schema-and-setting adaptation | Paired implementation comparison, not source-study replication |
| Ali XGB-125D | R02, R09 | Same-split reproduction | Paired but not independent evidence because the ComPlat lineage is shared |
| Bhattacharya--Roy interaction model | R01, R02, R03, R04, R05, R09 | Independently retrained adaptation | Primary split-matched adaptation; source-domain performance was not reproduced |
| Bhattacharya--Roy no-interaction model | R01, R02, R03, R04, R05, R09 | Prespecified architecture control | Control only; not a source-domain reproduction |
| Consensus GNN | R01, R02, R03, R04, R05, R09 | Independently retrained official-code adaptation | Split-matched adaptation, not the authors' released checkpoints |
| Consensus GNN | R05 | Authors' released predictions | Direct same-row comparison on the authors' 980-row test |

## Recorded execution environments

The provenance registry records the environments used for the evaluated adaptations. In particular, the Consensus GNN adaptation used Python 3.11.8, TensorFlow and Keras 2.15.0, the final pre-2.8.0 DeepChem 2.7.2 development distribution, RDKit 2023.09.6, and Open Babel 3.1.1. Environment identity is part of the comparator provenance because framework and chemistry-toolkit versions can affect features and predictions.

## What the hashes establish

The file-level hashes bind the reported comparisons to the inspected source and execution artefacts. They do not grant redistribution rights, replace the upstream resource, or make an adaptation an exact reproduction of an authors' training population. Users must obtain third-party resources from their upstream locations and follow their terms.
