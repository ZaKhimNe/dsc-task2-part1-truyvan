from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


PRODUCTION_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_ROOT = PRODUCTION_ROOT / "kaggle"
sys.path.insert(0, str(KAGGLE_ROOT))

from build import (  # noqa: E402
    build_input_dataset,
    build_kernel,
    bundle_sources,
    load_settings,
)


class ProductionKaggleBuildTests(unittest.TestCase):
    def test_settings_select_distinct_production_kernel(self):
        settings = load_settings(KAGGLE_ROOT / "settings.example.json")
        self.assertEqual(settings["model_version"], "v8_3_full_hard")
        self.assertIn("production", settings["kernel_slug"])
        self.assertNotEqual(settings["kernel_slug"], "dsc2026-legalqa-qlora-v83-full")

    def test_input_dataset_contains_configured_member_b_json(self):
        settings = load_settings(KAGGLE_ROOT / "settings.example.json")
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "input-dataset"
            build_input_dataset(settings, target_root=target)
            selected = target / settings["input_json"]
            metadata = json.loads(
                (target / "dataset-metadata.json").read_text(encoding="utf-8")
            )
            self.assertTrue(selected.is_file())
            self.assertEqual(metadata["id"], settings["input_dataset_slug"])

    def test_source_bundle_contains_selected_version_and_production_runner(self):
        settings = load_settings(KAGGLE_ROOT / "settings.example.json")
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "sources.zip"
            archive_path.write_bytes(bundle_sources(settings))
            with zipfile.ZipFile(archive_path) as archive:
                names = set(archive.namelist())
            self.assertIn("legalQA_Task2/scripts/run_inference.py", names)
            self.assertIn(
                "legalQA_Task2/src/legalqa_baseline/generation.py",
                names,
            )
            self.assertIn("legalQA_Task2/version.json", names)

    def test_kernel_metadata_attaches_input_and_adapter_datasets(self):
        settings = load_settings(KAGGLE_ROOT / "settings.example.json")
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "kernel"
            build_kernel(settings, target_root=target)
            metadata = json.loads(
                (target / "kernel-metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["dataset_sources"],
                [settings["input_dataset_slug"], settings["adapter_slug"]],
            )
            runner = (target / metadata["code_file"]).read_text(encoding="utf-8")
            self.assertNotIn("__SOURCE_BUNDLE__", runner)
            self.assertNotIn("__RUNTIME_SETTINGS__", runner)


if __name__ == "__main__":
    unittest.main()
