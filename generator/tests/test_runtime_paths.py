from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalqa_baseline.runtime_paths import resolve_input_dataset_root  # noqa: E402


class RuntimePathTests(unittest.TestCase):
    def test_resolves_legacy_slug_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            input_root = Path(directory)
            dataset_root = input_root / "legalqa-data"
            dataset_root.mkdir()
            (dataset_root / "validation.json").write_text("[]", encoding="utf-8")

            resolved = resolve_input_dataset_root(
                input_root,
                "owner/legalqa-data",
                "validation.json",
            )

            self.assertEqual(resolved, dataset_root)

    def test_resolves_nested_owner_and_slug_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            input_root = Path(directory)
            dataset_root = input_root / "datasets" / "owner" / "legalqa-data"
            dataset_root.mkdir(parents=True)
            (dataset_root / "validation.json").write_text("[]", encoding="utf-8")

            resolved = resolve_input_dataset_root(
                input_root,
                "owner/legalqa-data",
                "validation.json",
            )

            self.assertEqual(resolved, dataset_root)

    def test_raises_with_visible_candidates_when_required_file_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            input_root = Path(directory)
            (input_root / "unrelated-dataset").mkdir()

            with self.assertRaisesRegex(FileNotFoundError, "unrelated-dataset"):
                resolve_input_dataset_root(
                    input_root,
                    "owner/legalqa-data",
                    "validation.json",
                )


if __name__ == "__main__":
    unittest.main()
