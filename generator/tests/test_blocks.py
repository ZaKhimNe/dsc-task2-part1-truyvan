from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalqa_baseline.blocks import (  # noqa: E402
    assemble_answer,
    baseline_eligibility,
    extract_citation_metadata,
    split_answer,
)
import legalqa_baseline.generation as generation_module  # noqa: E402
from legalqa_baseline.generation import (  # noqa: E402
    ParsedOutput,
    conclusion_quality_issues,
    parse_model_output,
    parse_targeted_output,
    resolve_generated_blocks,
    should_accept_quality_retry,
)
from legalqa_baseline.fewshot import FewShotSelector, question_type  # noqa: E402
from legalqa_baseline.prompts import (  # noqa: E402
    INITIAL_SYSTEM_PROMPT,
    build_fallback_lead,
    build_initial_user_prompt,
    build_v8_sft_user_prompt,
    build_user_prompt,
)


class BlockTests(unittest.TestCase):
    def test_split_marker(self):
        answer = "Căn cứ Điều 1:\nNội dung luật.\nTheo đó, đây là kết luận."
        blocks = split_answer(answer)
        self.assertEqual(blocks.lead, "Căn cứ Điều 1:")
        self.assertEqual(blocks.quotation, "Nội dung luật.")
        self.assertEqual(blocks.conclusion, "Theo đó, đây là kết luận.")
        self.assertEqual(blocks.confidence, "high")
        self.assertEqual(
            assemble_answer(blocks.lead, blocks.quotation, blocks.conclusion), answer
        )

    def test_no_marker_is_low_confidence(self):
        blocks = split_answer("Căn cứ Điều 1:\nNội dung luật.")
        self.assertEqual(blocks.confidence, "low")
        self.assertEqual(blocks.conclusion, "")

    def test_parse_tagged_generation(self):
        result = parse_model_output(
            "<LEAD>Theo Điều 1:</LEAD>\n<CONCLUSION>Như vậy, được phép.</CONCLUSION>"
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.lead, "Theo Điều 1:")

    def test_partial_conclusion_is_preserved(self):
        result = parse_model_output(
            "<CONCLUSION>Như vậy, người này được phép.</CONCLUSION>"
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.lead, "")
        self.assertEqual(result.conclusion, "Như vậy, người này được phép.")
        self.assertEqual(result.method, "partial_xml_tags")

    def test_retry_lead_is_combined_with_initial_conclusion(self):
        result = resolve_generated_blocks(
            ParsedOutput("", "Kết luận ban đầu.", False, "partial_xml_tags"),
            ParsedOutput("Căn cứ Điều 1:", "", False, "partial_xml_tags"),
            None,
            {},
        )
        self.assertEqual(result["lead"], "Căn cứ Điều 1:")
        self.assertEqual(result["conclusion"], "Kết luận ban đầu.")
        self.assertTrue(result["model_blocks_complete"])
        self.assertEqual(result["lead_source"], "model_targeted_retry")

    def test_quality_retry_replaces_initial_conclusion(self):
        result = resolve_generated_blocks(
            ParsedOutput("Căn cứ Điều 1:", "Theo quy định như trên.", True, "xml_tags"),
            None,
            ParsedOutput("", "Có hai bước: bước một và bước hai.", True, "targeted_xml_tags"),
            {},
            prefer_retry_targets={"conclusion"},
        )
        self.assertEqual(result["conclusion"], "Có hai bước: bước một và bước hai.")
        self.assertEqual(result["conclusion_source"], "model_quality_retry")

    def test_quality_gate_detects_vague_short_list(self):
        issues = conclusion_quality_issues(
            "Thủ tục gồm những gì?",
            "1. Nộp hồ sơ.\n2. Kiểm tra hồ sơ.\n3. Trả kết quả.",
            "Thủ tục thực hiện theo quy định như trên.",
        )
        self.assertIn("vague_reference", issues)
        self.assertIn("short_list_answer", issues)

    def test_quality_gate_detects_missing_time_facts(self):
        issues = conclusion_quality_issues(
            "Thời hạn giải quyết là bao nhiêu?",
            "Thông thường là 10 ngày; trường hợp đặc biệt là 20 ngày.",
            "Thời hạn giải quyết là 10 ngày.",
        )
        self.assertIn("missing_numeric_facts", issues)

    def test_v8_prompt_is_question_type_aware(self):
        prompt = build_v8_sft_user_prompt(
            "Thủ tục bao gồm những gì?",
            "1. Bước một.\n2. Bước hai.",
            {},
        )
        self.assertIn("Liệt kê đầy đủ từng bước", prompt)
        self.assertIn("không thay nội dung", prompt)

    def test_v83_full_enables_classifier_c(self):
        self.assertEqual(
            question_type("Hồ sơ gồm những nội dung nào?"),
            "procedure_or_list",
        )
        self.assertEqual(
            question_type("Phiên họp xem xét quyết định những vấn đề gì?"),
            "procedure_or_list",
        )
        self.assertEqual(
            question_type("Người bệnh được hưởng những quyền lợi nào?"),
            "rights_and_duties",
        )
        self.assertEqual(
            question_type("Quyền và trách nhiệm của thành viên là gì?"),
            "rights_and_duties",
        )

    def test_v83_full_uses_classifier_c_prompt_for_rights_question(self):
        prompt = build_v8_sft_user_prompt(
            "Quyền và trách nhiệm của thành viên là gì?",
            "Thành viên có quyền biểu quyết và có trách nhiệm chấp hành điều lệ.",
            {},
        )
        self.assertIn("không được bỏ sót một phía", prompt)

    def test_v83b_prompt_disables_citation_guard_d(self):
        prompt = build_v8_sft_user_prompt(
            "Ai có thẩm quyền?",
            "Cơ quan có thẩm quyền thực hiện nhiệm vụ.",
            {"articles": ["Điều 14"], "clauses": ["khoản 3"]},
        )
        compact = " ".join(prompt.casefold().split())
        self.assertNotIn("không có tên hoặc số hiệu văn bản", compact)
        self.assertNotIn("không được suy đoán", compact)
        self.assertIn("viết khối 1 là câu dẫn tự nhiên", compact)

    def test_truncated_conclusion_at_token_limit_requests_expanded_retry(self):
        helper = getattr(
            generation_module,
            "should_expand_conclusion_retry",
            None,
        )
        self.assertTrue(
            callable(helper),
            "V8.3B cần có bộ phát hiện retry Khối 3 bị cắt do token limit",
        )
        if not callable(helper):
            return
        raw_output = "<CONCLUSION>1. Mục một. 2. Mục hai."
        parsed = parse_targeted_output(raw_output, "conclusion")
        self.assertFalse(parsed.valid)
        self.assertTrue(
            helper(
                raw_output=raw_output,
                parsed=parsed,
                output_tokens=160,
                max_new_tokens=160,
            )
        )

    def test_invalid_retry_below_token_limit_does_not_expand(self):
        helper = getattr(
            generation_module,
            "should_expand_conclusion_retry",
            None,
        )
        self.assertTrue(callable(helper))
        if not callable(helper):
            return
        raw_output = "<CONCLUSION>1. Mục một. 2. Mục hai."
        parsed = parse_targeted_output(raw_output, "conclusion")
        self.assertFalse(
            helper(
                raw_output=raw_output,
                parsed=parsed,
                output_tokens=159,
                max_new_tokens=160,
            )
        )

    def test_complete_retry_at_token_limit_does_not_expand(self):
        helper = getattr(
            generation_module,
            "should_expand_conclusion_retry",
            None,
        )
        self.assertTrue(callable(helper))
        if not callable(helper):
            return
        raw_output = "<CONCLUSION>Đây là kết luận đầy đủ.</CONCLUSION>"
        parsed = parse_targeted_output(raw_output, "conclusion")
        self.assertTrue(parsed.valid)
        self.assertFalse(
            helper(
                raw_output=raw_output,
                parsed=parsed,
                output_tokens=160,
                max_new_tokens=160,
            )
        )

    def test_adaptive_retry_summary_reports_usage_recovery_and_acceptance(self):
        helper = getattr(
            generation_module,
            "summarize_adaptive_retries",
            None,
        )
        self.assertTrue(callable(helper))
        if not callable(helper):
            return
        summary = helper(
            [
                {
                    "token_limit_retry_used": True,
                    "token_limit_retry_recovered": True,
                    "expanded_retry_accepted": True,
                },
                {
                    "token_limit_retry_used": True,
                    "token_limit_retry_recovered": False,
                    "expanded_retry_accepted": False,
                },
                {
                    "token_limit_retry_used": False,
                    "token_limit_retry_recovered": False,
                    "expanded_retry_accepted": False,
                },
                {
                    "token_limit_retry_used": False,
                    "token_limit_retry_recovered": False,
                    "expanded_retry_accepted": False,
                },
            ]
        )
        self.assertEqual(
            summary,
            {
                "token_limit_retry_rate": 0.5,
                "token_limit_retry_recovered_rate": 0.5,
                "expanded_retry_acceptance_rate": 0.5,
            },
        )

    def test_soft_gate_accepts_concise_candidate_with_better_coverage(self):
        helper = getattr(generation_module, "soft_retry_decision", None)
        self.assertTrue(callable(helper))
        if not callable(helper):
            return
        decision = helper(
            question="Hồ sơ gồm những gì?",
            context="1. Đơn đề nghị.\n2. Bản sao giấy tờ.\n3. Văn bản xác nhận.",
            initial_conclusion="Hồ sơ gồm các nội dung nêu trên.",
            retry_conclusion=(
                "Hồ sơ gồm: 1. Đơn đề nghị; 2. Bản sao giấy tờ; "
                "3. Văn bản xác nhận."
            ),
        )
        self.assertTrue(decision["accepted"])
        self.assertGreater(decision["retry_score"], decision["initial_score"])

    def test_soft_gate_rejects_candidate_that_changes_polarity(self):
        helper = getattr(generation_module, "soft_retry_decision", None)
        self.assertTrue(callable(helper))
        if not callable(helper):
            return
        decision = helper(
            question="Doanh nghiệp được cấp phép trong trường hợp nào?",
            context="1. Có giấy phép đầu tư.\n2. Đáp ứng điều kiện về vốn.",
            initial_conclusion="Doanh nghiệp được cấp phép khi đáp ứng các điều kiện.",
            retry_conclusion=(
                "Doanh nghiệp không được cấp phép dù có giấy phép đầu tư và "
                "đáp ứng điều kiện về vốn."
            ),
        )
        self.assertFalse(decision["accepted"])
        self.assertIn("polarity_changed", decision["safety_reasons"])

    def test_soft_gate_rejects_overly_verbose_sparse_candidate(self):
        helper = getattr(generation_module, "soft_retry_decision", None)
        self.assertTrue(callable(helper))
        if not callable(helper):
            return
        context = "\n".join(f"{index}. Mục hồ sơ {index}." for index in range(1, 16))
        retry = (
            "Hồ sơ gồm bảy mục có liên quan. "
            + " Nội dung giải thích bổ sung không trực tiếp cần thiết." * 70
        )
        decision = helper(
            question="Hồ sơ phải chuẩn bị như thế nào?",
            context=context,
            initial_conclusion="Hồ sơ chuẩn bị theo quy định nêu trên.",
            retry_conclusion=retry,
        )
        self.assertFalse(decision["accepted"])
        self.assertIn("score_margin_not_met", decision["reasons"])

    def test_soft_gate_summary_reports_acceptance_and_changed_output(self):
        helper = getattr(generation_module, "summarize_soft_gate", None)
        self.assertTrue(callable(helper))
        if not callable(helper):
            return
        summary = helper(
            [
                {
                    "soft_quality_retry_accepted": True,
                    "conclusion": "Kết luận hard.",
                    "soft_conclusion": "Kết luận soft tốt hơn.",
                },
                {
                    "soft_quality_retry_accepted": False,
                    "conclusion": "Giữ nguyên.",
                    "soft_conclusion": "Giữ nguyên.",
                },
            ]
        )
        self.assertEqual(
            summary,
            {
                "soft_gate_acceptance_rate": 0.5,
                "soft_gate_changed_rate": 0.5,
            },
        )

    def test_plain_text_is_rejected_for_targeted_conclusion(self):
        result = parse_targeted_output(
            "Như vậy, người này được phép thực hiện.", "conclusion"
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.conclusion, "")
        self.assertEqual(result.method, "targeted_missing_wrapper")

    def test_targeted_conclusion_rejects_stray_closing_tag(self):
        result = parse_targeted_output(
            "Không được</CONCLUSION>Như vậy, người này được phép.",
            "conclusion",
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.conclusion, "")
        self.assertEqual(result.method, "targeted_malformed_tags")

    def test_targeted_conclusion_accepts_one_exact_wrapper(self):
        result = parse_targeted_output(
            "<CONCLUSION>Như vậy, người này được phép.</CONCLUSION>",
            "conclusion",
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.conclusion, "Như vậy, người này được phép.")
        self.assertEqual(result.method, "targeted_xml_tags")

    def test_quality_retry_is_rejected_for_yes_no_question(self):
        accepted, reasons = should_accept_quality_retry(
            "Người lao động có được nghỉ hưởng chế độ thai sản không?",
            "Người lao động được nghỉ việc hưởng chế độ thai sản.",
            "Người lao động được nghỉ việc hưởng chế độ thai sản.",
            "Người lao động không được hưởng chế độ thai sản.",
        )
        self.assertFalse(accepted)
        self.assertIn("yes_no_guard", reasons)

    def test_quality_retry_rejects_candidate_that_still_has_quality_issue(self):
        accepted, reasons = should_accept_quality_retry(
            "Hồ sơ gồm những gì?",
            "1. Đơn đề nghị.\n2. Bản sao giấy tờ.\n3. Văn bản xác nhận.",
            "Hồ sơ thực hiện theo quy định nêu trên.",
            "Hồ sơ gồm những nội dung nêu trên.",
        )
        self.assertFalse(accepted)
        self.assertIn("quality_not_improved", reasons)

    def test_quality_retry_rejects_lost_numeric_fact(self):
        accepted, reasons = should_accept_quality_retry(
            "Khi nào được đơn phương chấm dứt hợp đồng?",
            "Người lao động tự ý nghỉ việc 05 ngày làm việc liên tục không có lý do.",
            "Không cần báo trước khi người lao động tự ý nghỉ việc 05 ngày liên tục.",
            "Không cần báo trước.",
        )
        self.assertFalse(accepted)
        self.assertIn("lost_numeric_fact", reasons)

    def test_quality_retry_detects_polarity_change_inside_negative_phrase(self):
        accepted, reasons = should_accept_quality_retry(
            "Doanh nghiệp được cấp phép trong trường hợp nào?",
            "1. Có giấy phép đầu tư.\n2. Đáp ứng điều kiện về vốn.",
            "Doanh nghiệp được cấp phép theo các điều kiện nêu trên.",
            (
                "Doanh nghiệp không được cấp phép dù đã có giấy phép đầu tư và "
                "đáp ứng đầy đủ điều kiện về vốn, nhân sự, địa điểm, phương án "
                "kinh doanh cùng toàn bộ yêu cầu pháp luật có liên quan."
            ),
        )
        self.assertFalse(accepted)
        self.assertIn("polarity_changed", reasons)

    def test_quality_retry_accepts_expanded_list_answer(self):
        accepted, reasons = should_accept_quality_retry(
            "Hồ sơ gồm những gì?",
            "1. Đơn đề nghị.\n2. Bản sao giấy tờ.\n3. Văn bản xác nhận.",
            "Hồ sơ thực hiện theo quy định nêu trên.",
            (
                "Hồ sơ gồm: 1. Đơn đề nghị; 2. Bản sao giấy tờ; "
                "3. Văn bản xác nhận của cơ quan có thẩm quyền."
            ),
        )
        self.assertTrue(accepted)
        self.assertEqual(reasons, [])

    def test_wrong_tag_is_rejected_for_targeted_conclusion(self):
        result = parse_targeted_output("<LEAD>Căn cứ Điều 1:</LEAD>", "conclusion")
        self.assertFalse(result.valid)
        self.assertEqual(result.conclusion, "")
        self.assertEqual(result.method, "targeted_wrong_tag")

    def test_targeted_conclusion_prompt_does_not_request_lead(self):
        prompt = build_user_prompt(
            "Ai được phép?",
            "Nội dung căn cứ.",
            {},
            target="conclusion",
        )
        self.assertIn("<CONCLUSION>...</CONCLUSION>", prompt)
        self.assertNotIn("<LEAD>...</LEAD>", prompt)

    def test_v7_initial_system_prompt_preserves_model_generated_lead_contract(self):
        self.assertEqual(
            INITIAL_SYSTEM_PROMPT,
            "Bạn là hệ thống sinh câu trả lời pháp luật tiếng Việt.\n"
            "Chỉ sử dụng căn cứ được cung cấp. Không tự thêm tên văn bản, Điều, Khoản,\n"
            "con số, điều kiện hoặc ngoại lệ không có trong câu hỏi và căn cứ.\n"
            "Bạn chỉ sinh Khối 1 và Khối 3; chương trình sẽ chép nguyên văn Khối 2.\n"
            "Bắt buộc trả về đủ cả thẻ <LEAD> và <CONCLUSION>.",
        )

    def test_v7_restores_v4_task_contract(self):
        prompt = build_initial_user_prompt(
            "Ai được phép?",
            "Nội dung căn cứ.",
            {},
        )
        compact = " ".join(prompt.split())
        self.assertIn("Khối 1", compact)
        self.assertIn("tối đa 60 từ", compact)
        self.assertIn("kết luận trực tiếp cho câu hỏi", compact)
        self.assertIn("tối đa 120 từ", compact)
        self.assertNotIn("tập nhỏ nhất", compact)

    def test_v7_targeted_conclusion_restores_v4_policy(self):
        prompt = build_user_prompt(
            "Ai được phép?",
            "Nội dung căn cứ.",
            {},
            target="conclusion",
        )
        compact = " ".join(prompt.split())
        self.assertIn("trực tiếp trả lời", compact)
        self.assertIn("tối đa 120 từ", compact)
        self.assertNotIn("tập nhỏ nhất", compact)

    def test_v7_prompt_contains_train_examples_before_new_question(self):
        example = {
            "question": "Ai thực hiện công việc?",
            "oracle_context": {
                "text": "Công việc do người có thẩm quyền thực hiện.",
                "citation_metadata": {"articles": ["Điều 1"]},
            },
            "target": {
                "lead": "Căn cứ Điều 1 quy định như sau:",
                "conclusion": "Như vậy, công việc do người có thẩm quyền thực hiện.",
            },
        }
        prompt = build_initial_user_prompt(
            "Câu hỏi mới?", "Căn cứ mới.", {}, fewshot_examples=[example]
        )
        self.assertIn("VÍ DỤ 1", prompt)
        self.assertIn("<LEAD>Căn cứ Điều 1 quy định như sau:</LEAD>", prompt)
        self.assertIn("Không dùng sự kiện", prompt)
        self.assertLess(prompt.index("VÍ DỤ 1"), prompt.index("Câu hỏi mới?"))

    def test_fewshot_selector_excludes_same_citation_group(self):
        def row(sample_id, group, question):
            return {
                "id": sample_id,
                "question": question,
                "citation_group": group,
                "baseline_eligible": True,
                "oracle_context": {
                    "text": "Nội dung pháp luật ngắn.",
                    "citation_metadata": {},
                },
                "target": {
                    "lead": "Căn cứ Điều 1 quy định như sau:",
                    "conclusion": "Như vậy, đây là kết luận.",
                },
            }

        selector = FewShotSelector(
            [
                row("train_same", "group-a", "Ai thực hiện?"),
                row("train_1", "group-b", "Ai có thẩm quyền?"),
                row("train_2", "group-c", "Đối tượng nào thực hiện?"),
            ]
        )
        query = row("validation", "group-a", "Ai thực hiện nhiệm vụ?")
        selected = selector.select(query, k=2)
        self.assertEqual({item["id"] for item in selected}, {"train_1", "train_2"})
        self.assertTrue(all(item["question_type"] == "subject" for item in selected))
        self.assertEqual(question_type("Thời hạn là bao nhiêu ngày?"), "number_or_time")

    def test_fallback_lead_uses_citation_metadata(self):
        lead = build_fallback_lead(
            {
                "clauses": ["khoản 1"],
                "articles": ["Điều 202"],
                "named_documents": ["Bộ luật Tố tụng Hình sự 2015"],
            }
        )
        self.assertEqual(
            lead,
            "Căn cứ theo khoản 1 Điều 202 Bộ luật Tố tụng Hình sự 2015, "
            "quy định liên quan như sau:",
        )

    def test_nested_quote_is_excluded_from_baseline(self):
        blocks = split_answer(
            "Căn cứ Điều 1:\nNội dung luật.\nTheo đó, kết luận.\nCăn cứ Điều 2:\nLuật khác."
        )
        eligible, reasons = baseline_eligibility(blocks)
        self.assertFalse(eligible)
        self.assertIn("nested_legal_quote_in_conclusion", reasons)

    def test_extract_citation_metadata(self):
        metadata = extract_citation_metadata(
            "Căn cứ khoản 1 Điều 202 Bộ luật Tố tụng Hình sự 2015 và Nghị định 90/2017/NĐ-CP:"
        )
        self.assertIn("khoản 1", [x.casefold() for x in metadata["clauses"]])
        self.assertIn("điều 202", [x.casefold() for x in metadata["articles"]])
        self.assertIn("90/2017/NĐ-CP", metadata["document_numbers"])


if __name__ == "__main__":
    unittest.main()
