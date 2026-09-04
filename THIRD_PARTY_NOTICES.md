# Third-party sources and notices

No third-party source tree, pretrained weight, or raw third-party dataset is bundled in the v1.0 package. The repository's MIT License applies only to original DLG-Sol source code and documentation and does not relicense any third-party resource listed below.

The study used or evaluated resources from the following upstream projects. Fixed revisions are recorded to identify the evaluated implementation; a revision is not a claim that all upstream data are redistributable.

| Resource | Upstream | Revision | Licence finding | Bundled here |
|---|---|---|---|---|
| PNNL solubility model | https://github.com/pnnl/solubility-prediction-paper | `8e22d552cd29d054df6dfcde76a2e52f39b455df` | Permissive Battelle terms; notice and disclaimer required for redistribution | No |
| Bhattacharya--Roy MLP--GNN | https://github.com/spriti523/MLP-GNN-for-Modeling-Aqueous-Solubility | `8c656265083fc5d9bc40beb118e73909deaa000d` | No repository licence identified | No |
| Consensus GNN | https://github.com/nadinulrich/log_Sw_prediction | `3da09dc96acde6c5c51d784307559067864ca779` | Repository MIT; included dataset has no separate data notice | No |
| Ali/ComPlat XGB-125D | https://github.com/ComPlat/water-solubility-prediction | `d4feda24b4bcb9efee9953605ba56af0e52dba4d` | README declares MIT; no LICENSE file at the pinned revision and upstream dataset rights are not established | No |
| AqSolDB | https://github.com/mcsorkun/AqSolDB | `8e02b548fd9a78778ff89a5aa9a460d1a289cc3a` | Repository code MIT; repository `data/` marked CC0-1.0 | No |
| Llompart et al. AqSolDBc dataset | https://doi.org/10.57745/CZVZIA | version 2.0, Dataverse file 569036 | Etalab Open Licence 2.0 | No |
| Therapeutics Data Commons software | https://github.com/mims-harvard/TDC | `c310c35f27e3f506411018ac43d97b8ba23ca652` | MIT software licence; dataset-specific rights require separate review | No |
| ChemBERTa base model | https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1 | `761d6a18cf99db371e0b43baf3e2d21b3e865a20` observed during the provenance audit; historical training revision not recorded | No explicit licence declared in the retrieved model metadata | No |
| Biogen pH 6.8 CLND supplementary panel | https://doi.org/10.1021/acs.jcim.3c00160 | Published article and associated public ADME data used in the supplementary analysis | No standalone row-level redistribution grant recorded for the study dataset | No |
| Ghanavati solubility repository | https://github.com/amingh1995/Aqueous-Solubility-Prediction | `0d0b5b609387ebd6b6a35ba14ba1c86cd7fe3c74` | No repository licence identified | No |

Users should obtain these resources from their upstream locations and comply with the applicable terms. The absence of a licence is treated conservatively and does not grant permission to copy or redistribute a resource.

The file-level identities and execution environments used for external-comparator adaptations are recorded in `configs/external_adapter_provenance.json`. Dataset and derived-asset bundling decisions are recorded separately in `configs/redistribution_decisions.json`; a decision not to bundle an openly licensed source is not a finding that its licence is restrictive.

## Runtime dependencies

The public descriptor implementation imports NumPy (BSD-3-Clause), pandas (BSD-3-Clause), scikit-learn (BSD-3-Clause), XGBoost (Apache-2.0), RDKit (BSD-3-Clause), and Mordred (BSD-3-Clause). The neural implementations additionally import PyTorch (BSD-3-Clause), Transformers (Apache-2.0), and PyTorch Geometric (MIT). These packages are dependencies only; their source code and binary distributions are not bundled in this repository.
