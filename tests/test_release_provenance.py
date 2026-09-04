from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import pandas as pd

from dlg_sol.provenance import validate_provenance_contract


ROOT = Path(__file__).resolve().parents[1]


def _records():
    external = json.loads((ROOT / "configs/external_adapter_provenance.json").read_text(encoding="utf-8"))
    redistribution = json.loads((ROOT / "configs/redistribution_decisions.json").read_text(encoding="utf-8"))
    table = pd.read_csv(ROOT / "results/table_2_absolute_rmse.csv").to_dict("records")
    return external, redistribution, table


class ReleaseProvenanceTests(unittest.TestCase):
    def test_release_contract_passes(self):
        self.assertEqual(validate_provenance_contract(*_records()), [])

    def test_unlicensed_source_cannot_be_bundled(self):
        external, redistribution, table = _records()
        external["families"][2]["bundled"] = True
        self.assertTrue(validate_provenance_contract(external, redistribution, table))

    def test_missing_table_comparator_is_rejected(self):
        external, redistribution, table = _records()
        external["families"][0]["model_records"] = []
        self.assertTrue(validate_provenance_contract(external, redistribution, table))

    def test_released_prediction_identity_drift_is_rejected(self):
        external, redistribution, table = _records()
        external["families"][3]["model_records"][1]["evidence_class"] = "independently retrained official-code adaptation"
        self.assertTrue(validate_provenance_contract(external, redistribution, table))

    def test_missing_dataset_decision_is_rejected(self):
        external, redistribution, table = _records()
        redistribution["sources"] = redistribution["sources"][:-1]
        self.assertTrue(validate_provenance_contract(external, redistribution, table))

    def test_private_path_is_rejected(self):
        external, redistribution, table = _records()
        external["families"][0]["file_evidence"][0]["logical_path"] = "/data/" + "koo/private.py"
        self.assertTrue(validate_provenance_contract(external, redistribution, table))

    def test_row_level_prediction_bundle_is_rejected(self):
        external, redistribution, table = _records()
        changed = copy.deepcopy(redistribution)
        changed["derived_asset_classes"][1]["bundled"] = True
        self.assertTrue(validate_provenance_contract(external, changed, table))


if __name__ == "__main__":
    unittest.main()
