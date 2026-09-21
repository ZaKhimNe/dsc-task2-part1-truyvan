from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PRODUCTION_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PRODUCTION_ROOT
sys.path.insert(0, str(PRODUCTION_ROOT / "src"))

from model_production.application import execute_run, load_config  # noqa: E402


class FakeGenerator:
    adapter_path = "fake-adapter"
    parameter_count = 123

    def generate(self, question, context, citation_metadata, seed, fewshot_examples=None):
        return {
            "lead": "Căn cứ quy định có liên quan:",
            "conclusion": "Theo đó, đây là kết luận.",
            "soft_conclusion": "Theo đó, đây là kết luận mềm.",
            "format_valid": True,
            "answer_complete": True,
            "retry_used": False,
            "quality_retry_used": False,
            "quality_retry_accepted": False,
            "token_limit_retry_used": False,
            "latency_seconds": 0.01,
            "attempts": [],
        }


class ProductionApplicationTests(unittest.TestCase):
    def test_load_config_rejects_unknown_model_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model_version": "missing_version",
                        "model_key": "qwen3_17b",
                        "input_json": "input.json",
                        "output_root": "outputs",
                        "top_k": 3,
                        "seed": 2026,
                        "limit": None,
                        "prompt_mode": "v8_sft",
                        "decoding_mode": "greedy",
                        "enable_quality_retry": True,
                        "adapter": {"local_path": "adapter", "kaggle_slug": "owner/adapter"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                load_config(path, workspace_root=WORKSPACE_ROOT)

    def test_execute_run_writes_answers_metrics_and_details(self):
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            input_path = temp / "input.json"
            output_root = temp / "outputs"
            config_path = temp / "production.json"
            input_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "147194",
                            "question": "Quy định là gì?",
                            "contexts": [
                                {
                                    "text": "Nội dung Điều 37.",
                                    "document_number": "59/2020/QH14",
                                    "article": "Điều 37",
                                    "clause": "",
                                    "retrieval_score": 0.9,
                                    "document": {"name": "Luật Doanh nghiệp 2020"},
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model_version": "v8_3_full_hard",
                        "model_key": "qwen3_17b",
                        "input_json": str(input_path),
                        "output_root": str(output_root),
                        "top_k": 1,
                        "seed": 2026,
                        "limit": None,
                        "prompt_mode": "v8_sft",
                        "decoding_mode": "greedy",
                        "enable_quality_retry": True,
                        "adapter": {"local_path": "fake-adapter", "kaggle_slug": "owner/adapter"},
                    }
                ),
                encoding="utf-8",
            )

            result = execute_run(
                config_path,
                workspace_root=WORKSPACE_ROOT,
                generator_factory=lambda **_: FakeGenerator(),
                run_id="test-run",
            )

            answers = json.loads((result / "answers.json").read_text(encoding="utf-8"))
            metrics = json.loads((result / "run_metrics.json").read_text(encoding="utf-8"))
            details = (result / "details.jsonl").read_text(encoding="utf-8")
            self.assertEqual(
                answers,
                {
                    "147194": {
                        "answer": "Căn cứ quy định có liên quan:\nNội dung Điều 37.\nTheo đó, đây là kết luận."
                    }
                },
            )
            self.assertEqual(metrics["model_version"], "v8_3_full_hard")
            self.assertEqual(metrics["sample_count"], 1)
            self.assertEqual(metrics["scoring_status"], "not_available_no_reference")
            self.assertIsNone(metrics["meteor"])
            self.assertIsNone(metrics["rouge_l"])
            self.assertIn('"id": "147194"', details)


if __name__ == "__main__":
    unittest.main()
