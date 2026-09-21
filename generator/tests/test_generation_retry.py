from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legalqa_baseline.generation import QwenGenerator


class GenerationRetryTests(unittest.TestCase):
    def test_missing_conclusion_expands_once_and_reports_selection(self):
        for final_text, expected_complete in (
            ("<CONCLUSION>Kết luận đầy đủ.</CONCLUSION>", True),
            ("<CONCLUSION>Vẫn bị cắt", False),
        ):
            with self.subTest(expected_complete=expected_complete):
                generator = QwenGenerator.__new__(QwenGenerator)
                generator.config = {
                    "max_input_tokens": 5120,
                    "max_new_tokens": 256,
                    "targeted_conclusion_max_new_tokens": 160,
                    "expanded_targeted_conclusion_max_new_tokens": 768,
                }
                generator.prompt_mode = "v8_sft"
                generator.decoding_mode = "greedy"
                generator.enable_quality_retry = False
                generator.adapter_path = None
                generator.device = "cpu"
                generator.torch = MagicMock()
                generator.torch.cuda.is_available.return_value = False
                generator._build_prompt = MagicMock(return_value=("user", "system"))
                generator._chat_text = MagicMock(return_value="prompt")
                encoded = MagicMock()
                encoded.__getitem__.return_value.shape = (1, 10)
                tokenizer_result = MagicMock()
                tokenizer_result.__getitem__.return_value = [1, 2]
                tokenizer_result.to.return_value = encoded
                generator.tokenizer = MagicMock(return_value=tokenizer_result)
                generator.tokenizer.decode.side_effect = [
                    "Căn cứ.",
                    "<LEAD>Căn cứ:</LEAD><CONCLUSION>Bị cắt",
                    "<CONCLUSION>Tiếp tục bị cắt",
                    final_text,
                ]
                generator.model = MagicMock()
                outputs = []
                for count in (256, 160, 20 if expected_complete else 768):
                    output = MagicMock()
                    output.__getitem__.return_value.shape = (count,)
                    outputs.append(output)
                generator.model.generate.side_effect = outputs

                result = generator.generate("Câu hỏi?", "Căn cứ.", {}, seed=2026)

                self.assertEqual(generator.model.generate.call_count, 3)
                self.assertEqual(
                    [call.kwargs["max_new_tokens"] for call in generator.model.generate.call_args_list],
                    [256, 160, 768],
                )
                self.assertTrue(result["token_limit_retry_used"])
                self.assertEqual(result["answer_complete"], expected_complete)
                self.assertEqual(result["token_limit_retry_recovered"], expected_complete)
                self.assertEqual(result["expanded_retry_accepted"], expected_complete)
                self.assertFalse(result["quality_retry_used"])
                self.assertFalse(result["quality_retry_accepted"])
                if expected_complete:
                    self.assertEqual(result["conclusion"], "Kết luận đầy đủ.")


if __name__ == "__main__":
    unittest.main()
