import json
import sys
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from model_production.application import _generate_shard, _select_gpu_count, execute_run


class SeedGenerator:
    parameter_count = 123
    adapter_path = "fake"

    def __init__(self, **kwargs):
        pass

    def generate(self, question, context, citation_metadata, seed, fewshot_examples):
        if question == "fail":
            raise RuntimeError("worker failed")
        return {"lead": "Lead:", "conclusion": str(seed), "format_valid": True,
                "answer_complete": True, "retry_used": False, "seed": seed}


def rows():
    return [{"id": str(i), "question": "Question?", "contexts": [{"text": "Context."}]}
            for i in range(5)]


class DualGpuTests(unittest.TestCase):
    def test_worker_batches_ten_and_keeps_last_partial_batch(self):
        sizes = []
        class BatchGenerator(SeedGenerator):
            def generate_batch(self, items):
                sizes.append(len(items))
                return [self.generate(**item) for item in items]
        indexed = [(i, dict(rows()[0], id=str(i))) for i in range(23)]
        result, _ = _generate_shard(indexed, {}, 1, 2026,
                                    generator_factory=BatchGenerator, batch_size=10)
        self.assertEqual(sizes, [10, 10, 3])
        self.assertEqual([r[1]["generation"]["seed"] for r in result], list(range(2026, 2049)))

    def test_device_selection(self):
        self.assertEqual([_select_gpu_count(None, n) for n in (0, 1, 2, 4)], [0, 1, 2, 2])
        self.assertEqual(_select_gpu_count(1, 2), 1)
        for requested, available in ((2, 1), (0, 2), (-1, 2), (True, 2)):
            with self.assertRaises(ValueError):
                _select_gpu_count(requested, available)

    def test_spawn_shards_preserve_original_seeds_and_results(self):
        indexed = list(enumerate(rows()))
        serial, _ = _generate_shard(indexed, {}, 1, 2026, generator_factory=SeedGenerator)
        with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(_generate_shard, indexed[i::2], {}, 1, 2026,
                                   generator_factory=SeedGenerator) for i in range(2)]
            parallel = sorted(item for future in futures for item in future.result()[0])
            self.assertEqual(parallel, serial)
            broken = rows()[0]
            broken["question"] = "fail"
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                pool.submit(_generate_shard, [(0, broken)], {}, 1, 2026,
                            generator_factory=SeedGenerator).result()
        self.assertEqual([item[1]["generation"]["seed"] for item in parallel], list(range(2026, 2031)))

    def test_gpu_binding_happens_before_model_loading(self):
        torch = MagicMock()
        def factory(**kwargs):
            torch.cuda.set_device.assert_called_once_with(1)
            return SeedGenerator()
        with patch.dict(sys.modules, {"torch": torch}):
            result, _ = _generate_shard([(0, rows()[0])], {}, 1, 2026,
                                        device_id=1, generator_factory=factory)
        self.assertTrue(result[0][1]["generation"]["answer_complete"])

    def test_execute_run_merges_both_devices_and_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = json.loads((ROOT / "configs/production.example.json").read_text(encoding="utf-8-sig"))
            (root / "input.json").write_text(json.dumps(rows()), encoding="utf-8")
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            torch = MagicMock()
            torch.cuda.device_count.return_value = 2
            # Execute worker bodies locally while checking the real orchestration.
            pool = MagicMock()
            def submit(fn, *args, **kwargs):
                future = MagicMock()
                future.result.return_value = fn(*args, **kwargs)
                return future
            pool.__enter__.return_value.submit.side_effect = submit
            with patch.dict(sys.modules, {"torch": torch}), patch(
                "model_production.application.ProcessPoolExecutor", return_value=pool
            ):
                out = execute_run(root / "config.json", workspace_root=ROOT,
                                  generator_factory=SeedGenerator, gpu_count=2,
                                  input_json=root / "input.json", output_root=root / "out", run_id="dual")
            self.assertEqual([call.kwargs["device_id"] for call in pool.__enter__.return_value.submit.call_args_list], [0, 1])
            answers = json.loads((out / "answers.json").read_text(encoding="utf-8"))
            self.assertEqual(list(answers), [str(i) for i in range(5)])
            metrics = json.loads((out / "run_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["gpu_count"], 2)
            self.assertEqual(metrics["worker_count"], 2)
            self.assertEqual(metrics["sample_count"], 5)
            self.assertEqual(metrics["parameter_count"], 123)


if __name__ == "__main__":
    unittest.main()
