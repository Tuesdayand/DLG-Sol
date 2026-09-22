"""Documentation and metadata regressions; no molecular data or training."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('release_validator_v105', ROOT / 'scripts/validate_release.py')
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class DocumentationReleaseTests(unittest.TestCase):
    def test_current_release_metadata(self):
        self.assertIn('version: 1.0.5\n', (ROOT / 'CITATION.cff').read_text())
        self.assertIn('**v1.0.5**', (ROOT / 'README.md').read_text())
        errors = []
        VALIDATOR.validate_release_policy(errors)
        self.assertEqual(errors, [])

    def test_historical_citation_record_preserved(self):
        record = json.loads((ROOT / 'results/verified_manifests/release_v1_0_4_validation.json').read_text())
        self.assertEqual(record['release_version'], '1.0.4')
        self.assertEqual(record['manuscript_alignment']['software_link_in_checked_manuscript'], 'v1.0.3')

    def test_snapshot_scope_is_explicit(self):
        text = (ROOT / 'docs/MANUSCRIPT_ALIGNMENT.md').read_text()
        self.assertIn('release_v1_0_5_validation.json', text)
        self.assertIn('specific file checks', text)
        self.assertIn('not replaced or moved', text)
        self.assertNotIn('revised submission package', text)

    def test_csv_annotation_mapping_is_retained(self):
        text = (ROOT / 'docs/MANUSCRIPT_ALIGNMENT.md').read_text()
        self.assertIn('0.614***', text)
        self.assertIn('0.614†', text)
        self.assertIn('four external models', text)
        self.assertIn('OCHEM', text)

    def test_release_scope_does_not_claim_new_experiments(self):
        notes = (ROOT / 'docs/RELEASE_NOTES_v1.0.5.md').read_text()
        self.assertIn('No model fitting', notes)
        self.assertIn('all 14 machine-readable result CSVs are unchanged', notes)
        self.assertIn('Historical validation records are retained unchanged', notes)

    def metadata_errors(self, old, new):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            for name in ['CITATION.cff', 'README.md', 'data/README.md',
                         'THIRD_PARTY_NOTICES.md', 'docs/V1_RELEASE_PLAN.md']:
                destination = temporary / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, destination)
            citation = temporary / 'CITATION.cff'
            citation.write_text(citation.read_text().replace(old, new))
            errors = []
            with patch.object(VALIDATOR, 'ROOT', temporary):
                VALIDATOR.validate_release_policy(errors)
            return errors

    def test_old_version_metadata_is_rejected(self):
        self.assertTrue(self.metadata_errors('version: 1.0.5', 'version: 1.0.4'))

    def test_wrong_release_date_is_rejected(self):
        self.assertTrue(self.metadata_errors('date-released: 2026-09-22', 'date-released: 2026-09-21'))


if __name__ == '__main__':
    unittest.main()
