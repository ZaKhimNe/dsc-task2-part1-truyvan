from __future__ import annotations

import sys
import unittest
from pathlib import Path


PRODUCTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PRODUCTION_ROOT / "src"))

from model_production.pipeline import (  # noqa: E402
    build_context_bundle,
    build_submission,
    resolve_model_version,
    score_if_available,
    validate_packages,
)


def sample_row(*, reference: str | None = "Đáp án chuẩn") -> dict:
    row = {
        "id": "111057",
        "question": "Nguyên tắc quản lý chất lượng là gì?",
        "contexts": [
            {
                "text": "Nội dung điều thứ nhất.",
                "document_number": "19/2013/TT-BYT",
                "article": "Điều 2",
                "clause": "khoản 1",
                "retrieval_score": 0.98,
                "document": {"name": "Thông tư về quản lý chất lượng"},
            },
            {
                "text": "Nội dung điều thứ hai.",
                "document_number": "40/2009/QH12",
                "article": "Điều 51",
                "clause": "khoản 2",
                "retrieval_score": 0.91,
                "document": {"title": "Luật Khám bệnh, chữa bệnh 2009"},
            },
        ],
    }
    if reference is not None:
        row["reference_answer"] = reference
    return row


class ProductionPipelineTests(unittest.TestCase):
    def test_validate_packages_rejects_duplicate_ids(self):
        rows = [sample_row(), sample_row()]
        with self.assertRaisesRegex(ValueError, "ID trùng"):
            validate_packages(rows)

    def test_build_context_bundle_uses_top_k_and_aggregates_citations(self):
        bundle = build_context_bundle(sample_row(), top_k=1)
        self.assertEqual(bundle["text"], "Nội dung điều thứ nhất.")
        self.assertEqual(bundle["citation_metadata"]["document_numbers"], ["19/2013/TT-BYT"])
        self.assertEqual(bundle["citation_metadata"]["articles"], ["Điều 2"])
        self.assertEqual(bundle["citation_metadata"]["clauses"], ["khoản 1"])
        self.assertEqual(bundle["citation_metadata"]["named_documents"], ["Thông tư về quản lý chất lượng"])
        self.assertEqual(bundle["selected_context_count"], 1)

    def test_build_submission_has_competition_shape(self):
        submission = build_submission(
            [{"id": "147194", "answer": "Theo quy định tại Điều 37..."}]
        )
        self.assertEqual(
            submission,
            {"147194": {"answer": "Theo quy định tại Điều 37..."}},
        )

    def test_score_if_available_uses_existing_scorer_when_all_references_exist(self):
        captured = {}

        def fake_scorer(prediction, reference):
            captured["prediction"] = prediction
            captured["reference"] = reference
            return {"meteor": 0.8, "rouge": 0.9, "count": 1, "per_sample": []}

        submission = {"111057": {"answer": "Dự đoán"}}
        result = score_if_available(submission, [sample_row()], scorer=fake_scorer)
        self.assertEqual(result["scoring_status"], "completed")
        self.assertEqual(result["meteor"], 0.8)
        self.assertEqual(result["rouge_l"], 0.9)
        self.assertEqual(captured["reference"], {"111057": "Đáp án chuẩn"})

    def test_score_if_available_reports_null_without_references(self):
        result = score_if_available(
            {"111057": {"answer": "Dự đoán"}},
            [sample_row(reference=None)],
            scorer=lambda *_: self.fail("Scorer không được gọi"),
        )
        self.assertEqual(result["scoring_status"], "not_available_no_reference")
        self.assertIsNone(result["meteor"])
        self.assertIsNone(result["rouge_l"])

    def test_resolve_model_version_finds_versioned_source_and_config(self):
        workspace = PRODUCTION_ROOT
        resolved = resolve_model_version(workspace, "v8_3_full_hard")
        self.assertTrue((resolved / "src" / "legalqa_baseline").is_dir())
        self.assertTrue((resolved / "configs" / "models.json").is_file())


if __name__ == "__main__":
    unittest.main()
