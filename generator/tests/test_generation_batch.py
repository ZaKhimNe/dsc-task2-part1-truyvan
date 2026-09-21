import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legalqa_baseline.generation import QwenGenerator


class BatchGenerationTests(unittest.TestCase):
    def test_batches_initial_and_retry_stages_without_losing_order(self):
        generator = QwenGenerator.__new__(QwenGenerator)
        generator.decoding_mode = "greedy"
        stages = []
        def steps(question, **kwargs):
            first = yield ({"question": question}, {"max_new_tokens": 256, "do_sample": False})
            if question == "retry":
                second = yield ({"question": question}, {"max_new_tokens": 160, "do_sample": False})
                third = yield ({"question": question}, {"max_new_tokens": 768, "do_sample": False})
                return (question, first, second, third)
            return (question, first)
        generator._generate_steps = steps
        def run(requests):
            stages.append([request[1]["max_new_tokens"] for request in requests])
            return [request[1]["max_new_tokens"] for request in requests]
        generator._run_batch_requests = run
        result = generator.generate_batch([{"question": "ok"}, {"question": "retry"}, {"question": "ok2"}])
        self.assertEqual(result, [("ok", 256), ("retry", 256, 160, 768), ("ok2", 256)])
        self.assertEqual(stages, [[256, 256, 256], [160], [768]])

    def test_batch_shrinking_mid_stage_keeps_every_item(self):
        generator = QwenGenerator.__new__(QwenGenerator)
        generator.decoding_mode = "greedy"
        generator._batch_limit = 4
        def steps(question):
            result = yield ({}, {"max_new_tokens": 256})
            return question
        generator._generate_steps = steps
        def run(requests):
            generator._batch_limit = 2
            return [None] * len(requests)
        generator._run_batch_requests = run
        self.assertEqual(generator.generate_batch([{"question": i} for i in range(10)]), list(range(10)))

    def test_tensor_padding_attention_mask_and_eos_token_counts(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch required for tensor integration check")
        generator = QwenGenerator.__new__(QwenGenerator)
        generator.torch = torch
        generator.tokenizer = MagicMock(eos_token_id=99)
        generator.model = MagicMock()
        generator.model.generation_config.eos_token_id = [98, 99]
        generator.model.generate.return_value = torch.tensor([
            [99, 11, 12, 7, 98, 99, 99],
            [21, 22, 23, 8, 9, 10, 11],
        ])
        requests = [({"input_ids": torch.tensor([[11, 12]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)}, {"max_new_tokens": 4}),
                    ({"input_ids": torch.tensor([[21, 22, 23]]), "attention_mask": torch.ones((1, 3), dtype=torch.long)}, {"max_new_tokens": 4})]
        result = generator._run_batch_once(requests)
        call = generator.model.generate.call_args.kwargs
        self.assertEqual(call["input_ids"].tolist(), [[99, 11, 12], [21, 22, 23]])
        self.assertEqual(call["attention_mask"].tolist(), [[0, 1, 1], [1, 1, 1]])
        self.assertEqual([tokens.tolist() for tokens, _ in result], [[7, 98], [8, 9, 10, 11]])

    def test_sampling_uses_serial_path_to_preserve_per_question_seeds(self):
        generator = QwenGenerator.__new__(QwenGenerator)
        generator.decoding_mode = "sample"
        generator.generate = MagicMock(side_effect=["a", "b"])
        self.assertEqual(generator.generate_batch([{"seed": 10}, {"seed": 11}]), ["a", "b"])
        self.assertEqual([call.kwargs["seed"] for call in generator.generate.call_args_list], [10, 11])

    def test_oom_splits_batch_and_does_not_swallow_single_sample_failure(self):
        generator = QwenGenerator.__new__(QwenGenerator)
        class OutOfMemoryError(RuntimeError):
            pass
        generator.torch = MagicMock()
        generator.torch.cuda.OutOfMemoryError = OutOfMemoryError
        generator._batch_limit = 10
        def run(requests):
            if len(requests) > 2:
                raise OutOfMemoryError("OOM")
            return requests
        generator._run_batch_once = run
        self.assertEqual(generator._run_batch_requests(list(range(5))), list(range(5)))
        self.assertLessEqual(generator._batch_limit, 2)
        generator._run_batch_once = MagicMock(side_effect=OutOfMemoryError("OOM"))
        with self.assertRaises(OutOfMemoryError):
            generator._run_batch_requests([1])


if __name__ == "__main__":
    unittest.main()
