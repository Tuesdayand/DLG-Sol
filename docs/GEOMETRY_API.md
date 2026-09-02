# Distance-aware three-dimensional graph branch

## Scope

Gate G1 exports the explicit-polar-hydrogen conformer graph and distance-aware scalar message-passing network used by the neural component. It does not export the four language–geometry fusion regressors, which form a separate release gate.

The historical implementation used an internal class name containing “EGNN.” The released architecture is described by its computation rather than that legacy name: atomic coordinates are converted into scalar distances and radial-basis edge features before message passing, and coordinates are not updated. The model is therefore a distance-aware three-dimensional message-passing neural network, not a coordinate-equivariant network.

## Conformer and graph construction

`build_geometry_graphs` applies the following recorded policy:

1. parse the supplied SMILES and retain the largest organic fragment;
2. attempt five ETKDGv3 conformers using the supplied record seed;
3. optimize with MMFF when parameters are available and otherwise use UFF;
4. rank by energy and retain at most three conformers;
5. keep heavy atoms and hydrogens bonded to nitrogen or oxygen as graph nodes;
6. add directed covalent edges and directed noncovalent proximity edges within 4.5 Å.

Each node has 42 features. Each edge has seven bond-status features, distance divided by 5 Å, and 16 Gaussian radial-basis values centered uniformly from 0 to 5 Å, for 24 features in total. Each conformer also carries the same 19 molecule-level auxiliary features for that record. Exact dimensions and constants are in `configs/geometry_branch.json`.

ETKDG and force-field results can vary across RDKit versions and platforms. The recorded RDKit version is pinned in `environment/geometry-requirements.txt`; the release does not claim bitwise coordinate identity across arbitrary environments.

## Model

`DistanceAware3DMPNN` first maps node and edge features to a common hidden dimension. Every layer constructs a message from the receiving node, sending node, and encoded edge, sums incoming messages, applies a residual node update with SiLU activations, and uses layer normalization. Global mean pooling and an embedding head produce one conformer representation. Representations and predictions are averaged arithmetically over retained conformers for each molecule.

The recorded search space contains two to four message-passing layers, hidden dimensions from 96 to 256, five dropout choices, three batch-size choices, log-uniform learning-rate and weight-decay ranges, and either the graph embedding alone or its concatenation with the 19 auxiliary features. Selection-partition RMSE is the checkpoint criterion. Dataset-specific candidate counts and transferred settings are preserved by their training adapters rather than being falsely represented as one universal checkpoint.

## Public API

Dependency-light configuration and feature utilities are exposed through `dlg_sol.geometry`:

- `load_geometry_config`;
- `largest_organic_fragment` and `generate_conformers`;
- `atom_feature_matrix`, `bond_base_features`, and `molecule_auxiliary_features`.

PyTorch Geometric functionality is in the following modules:

- `dlg_sol.geometry.graph`: `build_geometry_graphs`, `graph_from_conformer`, and `GeometryBuildResult`;
- `dlg_sol.geometry.modeling`: `DistanceAware3DMPNN`, `DistanceAwareMessagePassingLayer`, and `load_geometry_checkpoint`;
- `dlg_sol.geometry.training`: `sample_geometry_parameters`, `fit_geometry_model`, and `predict_geometry`.

Install `environment/geometry-requirements.txt` before importing these modules. The repository-wide environment is `environment/release-test-requirements.txt`.

Most archived checkpoints contain their model parameters. A legacy checkpoint that stores only dimensions and a state dictionary must be loaded with an explicit parameter mapping; the loader refuses to infer an unrecorded dropout or pooling policy from tensor shapes.

## Label firewall and failure boundary

Graph construction accepts molecular structure, a record identifier, configuration, and seed offset; it has no experimental-label argument. `predict_geometry` rejects scored graph objects that contain a `y` attribute. Fit and selection labels enter only through `fit_geometry_model`, and selection RMSE is used only during development-side checkpoint selection.

`GeometryBuildResult` reports parse, embedding, and graph failures explicitly. Gate G1 does not silently impute failed scored rows. Training-derived fallback construction and language–geometry fusion are handled at later release gates so that their provenance can be audited separately.

## Redistribution boundary

This gate does not redistribute molecular rows, experimental labels, conformer caches, coordinates, checkpoints, embeddings, or predictions. Its parity manifest records hashes, dimensions, aggregate differences, and protocol facts only. A fixed probe reproduced archived graph tensors exactly and reproduced archived embeddings and predictions within a `3e-6` absolute tolerance.
