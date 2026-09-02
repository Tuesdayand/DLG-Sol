from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem

from dlg_sol.geometry import atom_feature_matrix, bond_base_features, load_geometry_config, molecule_auxiliary_features


ROOT = Path(__file__).resolve().parents[1]
HAS_GEOMETRY_STACK = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torch_geometric") is not None


class GeometryContractTests(unittest.TestCase):
    def test_configuration_contract(self):
        config = load_geometry_config(ROOT / "configs" / "geometry_branch.json")
        self.assertEqual((config.node_dimension, config.edge_dimension, config.auxiliary_dimension), (42, 24, 19))
        self.assertEqual((config.conformers_attempted, config.conformers_retained), (5, 3))
        self.assertEqual(config.proximity_cutoff_angstrom, 4.5)

    def test_configuration_rejects_coordinate_update_claim(self):
        source = json.loads((ROOT / "configs" / "geometry_branch.json").read_text(encoding="utf-8"))
        source["model"]["coordinate_updates"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_geometry_config(path)

    def test_node_and_auxiliary_feature_dimensions(self):
        original = Chem.MolFromSmiles("CCO")
        molecule = Chem.AddHs(original)
        nodes = atom_feature_matrix(molecule)
        auxiliary = molecule_auxiliary_features(original, original, 1, 1.0, False, True, [0.0, 1.0])
        self.assertEqual(nodes.shape, (9, 42))
        self.assertEqual(auxiliary.shape, (19,))
        self.assertTrue(np.isfinite(nodes).all())
        self.assertTrue(np.isfinite(auxiliary).all())

    def test_bond_features(self):
        molecule = Chem.MolFromSmiles("CC")
        self.assertEqual(len(bond_base_features(molecule.GetBondWithIdx(0))), 7)
        self.assertEqual(bond_base_features(None), [0.0] * 7)

    @unittest.skipUnless(HAS_GEOMETRY_STACK, "geometry modelling dependencies are unavailable")
    def test_explicit_polar_hydrogen_graph(self):
        from dlg_sol.geometry.graph import build_geometry_graphs

        config = load_geometry_config(ROOT / "configs" / "geometry_branch.json")
        result = build_geometry_graphs("CCO", "ethanol", config)
        self.assertEqual(result.status, "ok")
        self.assertGreaterEqual(len(result.graphs), 1)
        graph = result.graphs[0]
        self.assertEqual(graph.x.shape[1], 42)
        self.assertEqual(graph.edge_attr.shape[1], 24)
        self.assertEqual(graph.aux.shape, (1, 19))
        self.assertEqual(graph.n_polar_h, 1)
        self.assertIsNone(graph.y)

    @unittest.skipUnless(HAS_GEOMETRY_STACK, "geometry modelling dependencies are unavailable")
    def test_parameter_sampler_and_label_free_inference(self):
        import torch

        from dlg_sol.geometry.graph import build_geometry_graphs
        from dlg_sol.geometry.modeling import DistanceAware3DMPNN
        from dlg_sol.geometry.training import predict_geometry, sample_geometry_parameters

        expected = {
            "hidden_dim": 256,
            "num_layers": 4,
            "dropout": 0.05,
            "batch_size": 64,
            "lr": 0.0004369919050718778,
            "weight_decay": 3.0088421993504876e-06,
            "pool": "mean_aux",
        }
        observed = sample_geometry_parameters(260705, 2)
        self.assertEqual(observed, expected)
        config = load_geometry_config(ROOT / "configs" / "geometry_branch.json")
        graphs = build_geometry_graphs("CCO", "ethanol", config).graphs
        model = DistanceAware3DMPNN(42, 24, 19, 32, 2, 0.05, "mean_aux")
        output = predict_geometry(model, graphs, 8, torch.device("cpu"))
        self.assertEqual(output.record_ids, ("ethanol",))
        self.assertEqual(output.predictions.shape, (1,))
        self.assertEqual(output.graph_embeddings.shape, (1, 32))
        self.assertEqual(output.auxiliary_features.shape, (1, 19))

    @unittest.skipUnless(HAS_GEOMETRY_STACK, "geometry modelling dependencies are unavailable")
    def test_scored_label_rejection(self):
        import torch

        from dlg_sol.geometry.graph import build_geometry_graphs
        from dlg_sol.geometry.modeling import DistanceAware3DMPNN
        from dlg_sol.geometry.training import predict_geometry

        config = load_geometry_config(ROOT / "configs" / "geometry_branch.json")
        graph = build_geometry_graphs("CCO", "ethanol", config).graphs[0]
        graph.y = torch.tensor([0.0])
        model = DistanceAware3DMPNN(42, 24, 19, 32, 2, 0.05, "mean_aux")
        with self.assertRaises(ValueError):
            predict_geometry(model, [graph], 8, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
