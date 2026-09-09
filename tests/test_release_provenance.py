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
    table = pd.read_csv(ROOT / "results/table_4_absolute_rmse.csv").to_dict("records")
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

    def test_article_table_3_source_context_matches_locked_contract(self):
        contract = json.loads(
            (ROOT / "configs/article_table_3_contract.json").read_text(encoding="utf-8")
        )
        source = pd.read_csv(
            ROOT / "supplementary/machine_readable/source_paper_benchmark_context.csv",
            keep_default_na=False,
        )
        expected_ids = [row["context_id"] for row in contract["rows"]]
        panel_ids = [context_id for panel in contract["panels"] for context_id in panel["context_ids"]]
        self.assertEqual(source.context_id.tolist(), expected_ids)
        self.assertEqual(expected_ids, panel_ids)
        self.assertEqual(len(source), 18)
        indexed = source.set_index("context_id", drop=False)
        for row in contract["rows"]:
            observed = indexed.loc[row["context_id"]]
            self.assertEqual(observed["model_or_method"], row["model_or_method"])
            self.assertEqual(int(observed["n"]), row["n"])
        for context_id, scope in contract["required_qualifiers"].items():
            self.assertEqual(indexed.loc[context_id, "comparison_scope"], scope)

    def test_article_table_4_matches_locked_contract(self):
        contract = json.loads(
            (ROOT / "configs/article_table_4_contract.json").read_text(encoding="utf-8")
        )
        table = pd.read_csv(ROOT / "results/table_4_absolute_rmse.csv", keep_default_na=False)
        indexed = table.set_index(["model_id", "variant", "panel"], drop=False)
        self.assertEqual(len(table), 30)
        for model in contract["models"]:
            for panel in (record["id"] for record in contract["panels"]):
                key = (model["model_id"], model["variant"], panel)
                row = indexed.loc[key]
                self.assertEqual(row["model_label"], model["model_label"])
                self.assertEqual(row["evidence_class"], model["evidence_class"])
                expected = model["display_rmse"][panel]
                self.assertEqual(bool(row["available"]), expected is not None)
                if expected is not None:
                    self.assertEqual(f"{float(row['absolute_rmse']):.3f}", expected)

    def test_bhattacharya_roy_primary_and_sensitivity_roles(self):
        source = pd.read_csv(
            ROOT / "supplementary/machine_readable/external_comparator_metrics_31_rows.csv"
        )
        additive = source[
            source.model_id.eq("bhattacharya_roy")
            & source.variant.eq("no_interaction")
        ]
        interaction = source[
            source.model_id.eq("bhattacharya_roy")
            & source.variant.eq("interaction")
        ]
        self.assertEqual(set(additive.main_table3_comparison), {True})
        self.assertEqual(set(interaction.main_table3_comparison), {False})
        self.assertEqual(set(additive.evidence_type), {"independently retrained architecture adaptation"})
        self.assertEqual(set(interaction.evidence_type), {"prespecified architecture sensitivity"})
        self.assertTrue(additive.delta_rmse_ci99_low.notna().all())
        self.assertTrue(interaction.delta_rmse_ci99_low.notna().all())


if __name__ == "__main__":
    unittest.main()
