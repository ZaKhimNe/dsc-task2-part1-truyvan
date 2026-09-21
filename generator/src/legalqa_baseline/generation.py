from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .blocks import assemble_answer
from .fewshot import content_tokens, question_type
from .prompts import (
    INITIAL_SYSTEM_PROMPT,
    TARGETED_RETRY_SYSTEM_PROMPT,
    V8_SFT_SYSTEM_PROMPT,
    build_fallback_lead,
    build_initial_user_prompt,
    build_targeted_retry_prompt,
    build_v8_sft_user_prompt,
)


LEAD_RE = re.compile(r"<LEAD>(?P<value>.*?)</LEAD>", re.IGNORECASE | re.DOTALL)
CONCLUSION_RE = re.compile(
    r"<CONCLUSION>(?P<value>.*?)</CONCLUSION>", re.IGNORECASE | re.DOTALL
)
VAGUE_CONCLUSION_RE = re.compile(
    r"\b(?:nêu trên|như trên|quy định trên|theo quy định nêu trên|các nội dung trên)\b",
    re.IGNORECASE,
)
LIST_ITEM_RE = re.compile(
    r"(?m)^\s*(?:[-–•]|\(?\d+[.)]|\(?[a-zđ][.)])\s+",
    re.IGNORECASE,
)
TIME_OR_AMOUNT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:ngày|tháng|năm|giờ|phút|%|phần trăm)\b",
    re.IGNORECASE,
)
PROTOCOL_TAG_RE = re.compile(r"</?(?:LEAD|CONCLUSION)\b[^>]*>", re.IGNORECASE)
NEGATIVE_POLARITY_RE = re.compile(
    r"\b(?:không được|không phải|không cần|không bắt buộc|bị cấm|không đủ)\b",
    re.IGNORECASE,
)
POSITIVE_POLARITY_RE = re.compile(
    r"\b(?:được|phải|cần|bắt buộc|có thể)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedOutput:
    lead: str
    conclusion: str
    valid: bool
    method: str


def parse_model_output(text: str) -> ParsedOutput:
    cleaned = text.strip()
    lead_match = LEAD_RE.search(cleaned)
    conclusion_match = CONCLUSION_RE.search(cleaned)
    if lead_match or conclusion_match:
        lead = lead_match.group("value").strip() if lead_match else ""
        conclusion = (
            conclusion_match.group("value").strip() if conclusion_match else ""
        )
        valid = bool(lead and conclusion)
        return ParsedOutput(
            lead,
            conclusion,
            valid,
            "xml_tags" if valid else "partial_xml_tags",
        )
    try:
        obj = json.loads(cleaned)
        lead = str(obj.get("lead", "")).strip()
        conclusion = str(obj.get("conclusion", "")).strip()
        valid = bool(lead and conclusion)
        return ParsedOutput(
            lead,
            conclusion,
            valid,
            "json" if valid else "partial_json",
        )
    except (json.JSONDecodeError, TypeError, AttributeError):
        return ParsedOutput("", "", False, "unparsed")


def parse_targeted_output(text: str, target: str) -> ParsedOutput:
    """Parse retry chỉ khi model trả về đúng một wrapper của khối yêu cầu."""

    if target not in {"lead", "conclusion"}:
        raise ValueError(f"Target retry không hợp lệ: {target}")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:xml|json|text)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    other_tag = "CONCLUSION" if target == "lead" else "LEAD"
    if re.search(rf"</?{other_tag}\b", cleaned, flags=re.IGNORECASE):
        return ParsedOutput("", "", False, "targeted_wrong_tag")

    target_tag = target.upper()
    wrapper = re.fullmatch(
        rf"<{target_tag}>\s*(?P<value>.*?)\s*</{target_tag}>",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not wrapper:
        method = (
            "targeted_malformed_tags"
            if PROTOCOL_TAG_RE.search(cleaned)
            else "targeted_missing_wrapper"
        )
        return ParsedOutput("", "", False, method)

    value = wrapper.group("value").strip()
    if PROTOCOL_TAG_RE.search(value):
        return ParsedOutput("", "", False, "targeted_malformed_tags")
    if not value:
        return ParsedOutput("", "", False, "targeted_empty")
    return ParsedOutput(
        value if target == "lead" else "",
        value if target == "conclusion" else "",
        True,
        "targeted_xml_tags",
    )


def should_expand_conclusion_retry(
    *,
    raw_output: str,
    parsed: ParsedOutput,
    output_tokens: int,
    max_new_tokens: int,
) -> bool:
    """Chỉ retry thêm khi Khối 3 bị cắt đúng tại giới hạn sinh token."""

    normalized = raw_output.casefold()
    return bool(
        not parsed.valid
        and output_tokens >= max_new_tokens
        and "<conclusion>" in normalized
        and "</conclusion>" not in normalized
    )


def summarize_adaptive_retries(
    generations: list[dict[str, Any]],
) -> dict[str, float]:
    """Tổng hợp mức kích hoạt, phục hồi và được gate chấp nhận của V8.3B."""

    total = len(generations)
    used = sum(bool(item.get("token_limit_retry_used")) for item in generations)
    recovered = sum(
        bool(item.get("token_limit_retry_recovered")) for item in generations
    )
    accepted = sum(bool(item.get("expanded_retry_accepted")) for item in generations)
    return {
        "token_limit_retry_rate": used / total if total else 0.0,
        "token_limit_retry_recovered_rate": recovered / used if used else 0.0,
        "expanded_retry_acceptance_rate": accepted / used if used else 0.0,
    }


def summarize_soft_gate(
    generations: list[dict[str, Any]],
) -> dict[str, float]:
    total = len(generations)
    accepted = sum(
        bool(item.get("soft_quality_retry_accepted")) for item in generations
    )
    changed = sum(
        item.get("soft_conclusion") != item.get("conclusion")
        for item in generations
    )
    return {
        "soft_gate_acceptance_rate": accepted / total if total else 0.0,
        "soft_gate_changed_rate": changed / total if total else 0.0,
    }


def conclusion_quality_issues(
    question: str,
    context: str,
    conclusion: str,
) -> list[str]:
    """Phát hiện Khối 3 có nguy cơ bỏ sót rõ ràng để quyết định retry.

    Đây là quality gate bảo thủ, không dùng đáp án chuẩn và không chấm điểm trong lúc
    inference. Nó chỉ nhìn câu hỏi, context mà model đã được phép thấy và đầu ra model.
    """

    if not conclusion.strip():
        return ["missing_conclusion"]
    issues: list[str] = []
    normalized_conclusion = " ".join(conclusion.split())
    word_count = len(normalized_conclusion.split())
    kind = question_type(question)

    if VAGUE_CONCLUSION_RE.search(normalized_conclusion):
        issues.append("vague_reference")

    context_items = len(LIST_ITEM_RE.findall(context))
    if kind == "procedure_or_list" and context_items >= 2 and word_count < 45:
        issues.append("short_list_answer")
    if kind == "condition" and context_items >= 2 and word_count < 40:
        issues.append("short_condition_answer")

    if kind == "number_or_time":
        context_facts = {
            " ".join(match.casefold().split())
            for match in TIME_OR_AMOUNT_RE.findall(context)
        }
        conclusion_facts = {
            " ".join(match.casefold().split())
            for match in TIME_OR_AMOUNT_RE.findall(normalized_conclusion)
        }
        required_fact_count = min(2, len(context_facts))
        if required_fact_count and len(conclusion_facts) < required_fact_count:
            issues.append("missing_numeric_facts")
    return issues


def _polarity(text: str) -> str | None:
    has_negative = bool(NEGATIVE_POLARITY_RE.search(text))
    text_without_negative_phrases = NEGATIVE_POLARITY_RE.sub(" ", text)
    has_positive = bool(POSITIVE_POLARITY_RE.search(text_without_negative_phrases))
    if has_negative and not has_positive:
        return "negative"
    if has_positive and not has_negative:
        return "positive"
    return None


def _numeric_facts(text: str) -> set[str]:
    return {
        " ".join(match.casefold().split())
        for match in TIME_OR_AMOUNT_RE.findall(text)
    }


def should_accept_quality_retry(
    question: str,
    context: str,
    initial_conclusion: str,
    retry_conclusion: str,
) -> tuple[bool, list[str]]:
    """Chọn retry mà không nhìn đáp án chuẩn và không làm mất dữ kiện đã có."""

    reasons: list[str] = []
    if not retry_conclusion.strip():
        return False, ["retry_empty"]

    kind = question_type(question)
    if kind == "yes_no":
        reasons.append("yes_no_guard")

    initial_issues = conclusion_quality_issues(
        question, context, initial_conclusion
    )
    retry_issues = conclusion_quality_issues(question, context, retry_conclusion)
    if not initial_issues:
        reasons.append("initial_not_flagged")
    elif len(retry_issues) >= len(initial_issues):
        reasons.append("quality_not_improved")

    initial_polarity = _polarity(initial_conclusion)
    retry_polarity = _polarity(retry_conclusion)
    if (
        initial_polarity
        and retry_polarity
        and initial_polarity != retry_polarity
    ):
        reasons.append("polarity_changed")

    lost_numeric_facts = _numeric_facts(initial_conclusion) - _numeric_facts(
        retry_conclusion
    )
    if lost_numeric_facts:
        reasons.append("lost_numeric_fact")

    if kind in {"procedure_or_list", "condition"}:
        initial_word_count = len(initial_conclusion.split())
        retry_word_count = len(retry_conclusion.split())
        if retry_word_count < initial_word_count:
            reasons.append("candidate_shorter")

    return not reasons, reasons


def _repeated_ngram_ratio(text: str, n: int = 4) -> float:
    tokens = [token.casefold() for token in re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", text)]
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def _soft_answer_score(
    question: str,
    context: str,
    answer: str,
    initial_conclusion: str,
) -> tuple[float, dict[str, float | int]]:
    issues = conclusion_quality_issues(question, context, answer)
    question_tokens = content_tokens(question)
    answer_tokens = content_tokens(answer)
    keyword_coverage = (
        len(question_tokens & answer_tokens) / len(question_tokens)
        if question_tokens
        else 0.0
    )

    context_items = len(LIST_ITEM_RE.findall(context))
    answer_items = len(LIST_ITEM_RE.findall(answer))
    list_coverage = min(1.0, answer_items / context_items) if context_items else 0.0

    context_numbers = _numeric_facts(context)
    answer_numbers = _numeric_facts(answer)
    numeric_coverage = (
        len(context_numbers & answer_numbers) / min(2, len(context_numbers))
        if context_numbers
        else 0.0
    )
    numeric_coverage = min(1.0, numeric_coverage)

    initial_tokens = content_tokens(initial_conclusion)
    retention = (
        len(initial_tokens & answer_tokens) / len(initial_tokens)
        if initial_tokens
        else 1.0
    )
    word_count = len(answer.split())
    word_budget = max(220, min(320, 18 * context_items))
    verbosity_ratio = max(0.0, (word_count - word_budget) / word_budget)
    repetition_ratio = _repeated_ngram_ratio(answer)

    score = (
        -4.0 * len(issues)
        + 3.0 * keyword_coverage
        + 3.0 * list_coverage
        + 2.0 * numeric_coverage
        + retention
        - 8.0 * verbosity_ratio
        - 3.0 * repetition_ratio
    )
    return score, {
        "quality_issue_count": len(issues),
        "keyword_coverage": round(keyword_coverage, 6),
        "list_coverage": round(list_coverage, 6),
        "numeric_coverage": round(numeric_coverage, 6),
        "initial_fact_retention": round(retention, 6),
        "word_count": word_count,
        "word_budget": word_budget,
        "verbosity_ratio": round(verbosity_ratio, 6),
        "repetition_ratio": round(repetition_ratio, 6),
    }


def soft_retry_decision(
    question: str,
    context: str,
    initial_conclusion: str,
    retry_conclusion: str,
    *,
    margin: float = 0.75,
) -> dict[str, Any]:
    """Chọn retry bằng score mềm sau các veto an toàn không dùng gold answer."""

    safety_reasons: list[str] = []
    if not retry_conclusion.strip():
        safety_reasons.append("retry_empty")
    if question_type(question) == "yes_no":
        safety_reasons.append("yes_no_guard")
    initial_polarity = _polarity(initial_conclusion)
    retry_polarity = _polarity(retry_conclusion)
    if initial_polarity and retry_polarity and initial_polarity != retry_polarity:
        safety_reasons.append("polarity_changed")
    if _numeric_facts(initial_conclusion) - _numeric_facts(retry_conclusion):
        safety_reasons.append("lost_numeric_fact")

    initial_score, initial_components = _soft_answer_score(
        question,
        context,
        initial_conclusion,
        initial_conclusion,
    )
    retry_score, retry_components = _soft_answer_score(
        question,
        context,
        retry_conclusion,
        initial_conclusion,
    )
    reasons = list(safety_reasons)
    if not safety_reasons and retry_score < initial_score + margin:
        reasons.append("score_margin_not_met")
    return {
        "accepted": not reasons,
        "initial_score": round(initial_score, 6),
        "retry_score": round(retry_score, 6),
        "margin": margin,
        "safety_reasons": safety_reasons,
        "reasons": reasons,
        "initial_components": initial_components,
        "retry_components": retry_components,
    }


def resolve_generated_blocks(
    initial: ParsedOutput,
    retry_lead: ParsedOutput | None,
    retry_conclusion: ParsedOutput | None,
    citation_metadata: dict[str, list[str]],
    *,
    prefer_retry_targets: set[str] | None = None,
) -> dict[str, str | bool]:
    prefer_retry_targets = prefer_retry_targets or set()
    retry_lead = retry_lead or ParsedOutput("", "", False, "not_used")
    retry_conclusion = retry_conclusion or ParsedOutput(
        "", "", False, "not_used"
    )
    if initial.lead:
        lead = initial.lead
        lead_source = "model_initial"
    elif retry_lead.lead:
        lead = retry_lead.lead
        lead_source = "model_targeted_retry"
    else:
        lead = build_fallback_lead(citation_metadata)
        lead_source = "template_fallback"

    if "conclusion" in prefer_retry_targets and retry_conclusion.conclusion:
        conclusion = retry_conclusion.conclusion
        conclusion_source = "model_quality_retry"
    elif initial.conclusion:
        conclusion = initial.conclusion
        conclusion_source = "model_initial"
    elif retry_conclusion.conclusion:
        conclusion = retry_conclusion.conclusion
        conclusion_source = "model_targeted_retry"
    else:
        conclusion = ""
        conclusion_source = "missing"

    return {
        "lead": lead,
        "conclusion": conclusion,
        "lead_source": lead_source,
        "conclusion_source": conclusion_source,
        "model_blocks_complete": bool(
            (initial.lead or retry_lead.lead)
            and (initial.conclusion or retry_conclusion.conclusion)
        ),
        "answer_complete": bool(lead and conclusion),
    }


class QwenGenerator:
    def __init__(
        self,
        config: dict[str, Any],
        cache_dir: str | Path | None = None,
        *,
        adapter_path: str | Path | None = None,
        prompt_mode: str = "v7_fewshot",
        decoding_mode: str = "sample",
        enable_quality_retry: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if prompt_mode not in {"v7_fewshot", "v8_sft"}:
            raise ValueError(f"prompt_mode không hợp lệ: {prompt_mode}")
        if decoding_mode not in {"sample", "greedy", "low_temperature"}:
            raise ValueError(f"decoding_mode không hợp lệ: {decoding_mode}")

        self.torch = torch
        self.config = config
        self.model_id = config["model_id"]
        self.family = config["family"]
        self.prompt_mode = prompt_mode
        self.decoding_mode = decoding_mode
        self.enable_quality_retry = enable_quality_retry
        self.adapter_path = str(Path(adapter_path).resolve()) if adapter_path else None
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, cache_dir=cache_dir, use_fast=True
        )
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if self.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    def _build_prompt(
        self,
        question: str,
        context: str,
        citation_metadata: dict[str, list[str]],
        *,
        target: str,
        fewshot_examples: list[dict] | None = None,
        legacy_retry: bool = False,
    ) -> tuple[str, str]:
        if self.prompt_mode == "v8_sft":
            if fewshot_examples:
                raise ValueError("Prompt V8 SFT không nhận ví dụ few-shot")
            return (
                build_v8_sft_user_prompt(
                    question,
                    context,
                    citation_metadata,
                    target=target,
                ),
                V8_SFT_SYSTEM_PROMPT,
            )
        if target == "both":
            return (
                build_initial_user_prompt(
                    question,
                    context,
                    citation_metadata,
                    legacy_retry=legacy_retry,
                    fewshot_examples=fewshot_examples,
                ),
                INITIAL_SYSTEM_PROMPT,
            )
        return (
            build_targeted_retry_prompt(
                question,
                context,
                citation_metadata,
                target=target,
            ),
            TARGETED_RETRY_SYSTEM_PROMPT,
        )

    def _chat_text(self, user_prompt: str, system_prompt: str) -> str:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.family == "qwen3":
            kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )

    def generate(self, question, context, citation_metadata, seed, fewshot_examples=None):
        steps = self._generate_steps(question, context, citation_metadata, seed, fewshot_examples)
        request = next(steps)
        while True:
            encoded, kwargs = request
            started = time.perf_counter()
            with self.torch.inference_mode():
                output = self.model.generate(**encoded, **kwargs)
            generated = output[0, encoded["input_ids"].shape[1]:]
            elapsed = time.perf_counter() - started
            try:
                request = steps.send((generated, elapsed))
            except StopIteration as completed:
                return completed.value

    def generate_batch(self, items):
        # Greedy production can batch safely; keep sampling's per-question RNG isolation.
        if self.decoding_mode != "greedy":
            return [self.generate(**item) for item in items]
        self._batch_limit = min(getattr(self, "_batch_limit", len(items)), len(items)) if items else 1
        steps = [self._generate_steps(**item) for item in items]
        pending = {i: next(step) for i, step in enumerate(steps)}
        results = [None] * len(items)
        while pending:
            groups = {}
            for index, request in pending.items():
                key = tuple(sorted(request[1].items()))
                groups.setdefault(key, []).append((index, request))
            pending = {}
            for group in groups.values():
                start = 0
                while start < len(group):
                    chunk = group[start:start + self._batch_limit]
                    outputs = self._run_batch_requests([request for _, request in chunk])
                    start += len(chunk)
                    if len(outputs) != len(chunk):
                        raise RuntimeError("Batch output count does not match input count")
                    for (index, _), output in zip(chunk, outputs):
                        try:
                            pending[index] = steps[index].send(output)
                        except StopIteration as completed:
                            results[index] = completed.value
        return results

    def _run_batch_requests(self, requests):
        try:
            return self._run_batch_once(requests)
        except self.torch.cuda.OutOfMemoryError:
            if len(requests) == 1:
                raise
            self._batch_limit = min(self._batch_limit, max(1, len(requests) // 2))
            print(f"CUDA OOM: reducing batch on {self.device if hasattr(self, 'device') else 'GPU'} "
                  f"to {self._batch_limit}", flush=True)
        # Release failed generate tensors before retrying smaller batches.
        import gc
        gc.collect()
        self.torch.cuda.empty_cache()
        middle = len(requests) // 2
        return (self._run_batch_requests(requests[:middle])
                + self._run_batch_requests(requests[middle:]))

    def _run_batch_once(self, requests):
        width = max(encoded["input_ids"].shape[1] for encoded, _ in requests)
        pad_id = self.tokenizer.eos_token_id
        encoded_batch = {}
        # Decoder-only generation must left-pad and mask padding, including EOS padding.
        for key, pad_value in (("input_ids", pad_id), ("attention_mask", 0)):
            encoded_batch[key] = self.torch.cat([
                self.torch.nn.functional.pad(encoded[key],
                    (width - encoded["input_ids"].shape[1], 0), value=pad_value)
                for encoded, _ in requests
            ], dim=0)
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**encoded_batch, **requests[0][1])
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        eos_ids.add(pad_id)
        generated = []
        for row in output[:, width:]:
            # Remove per-sequence padding so short answers don't look token-limited.
            ids = row.tolist()
            length = next((i + 1 for i, token in enumerate(ids) if token in eos_ids), len(ids))
            generated.append(row[:length])
        elapsed = (time.perf_counter() - started) / len(requests)
        return [(tokens, elapsed) for tokens in generated]

    def _generate_steps(
        self,
        question: str,
        context: str,
        citation_metadata: dict[str, list[str]],
        seed: int,
        fewshot_examples: list[dict] | None = None,
    ):
        max_input_tokens = int(self.config["max_input_tokens"])
        base_prompt_max_input_tokens = int(
            self.config.get("base_prompt_max_input_tokens", max_input_tokens)
        )
        # Context của câu hiện tại dùng đúng budget 3072-token của V4. Cửa sổ
        # tổng chỉ được mở rộng để đặt ví dụ train, không để V7 thấy thêm context.
        empty_prompts = []
        for legacy_retry in (False, True):
            empty_user_prompt, empty_system_prompt = self._build_prompt(
                question,
                "",
                citation_metadata,
                target="both",
                legacy_retry=legacy_retry,
            )
            empty_prompts.append(self._chat_text(empty_user_prompt, empty_system_prompt))
        fixed_token_count = max(
            len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            for prompt in empty_prompts
        )
        context_budget = max(
            128, base_prompt_max_input_tokens - fixed_token_count - 16
        )
        context_ids = self.tokenizer(
            context,
            add_special_tokens=False,
            truncation=True,
            max_length=context_budget,
        )["input_ids"]
        visible_context = self.tokenizer.decode(context_ids, skip_special_tokens=True)
        attempts = []
        targets = ["both"]
        for attempt_index, requested_target in enumerate(targets):
            expanded_token_retry = requested_target == "conclusion_expanded"
            target = "conclusion" if expanded_token_retry else requested_target
            user_prompt, system_prompt = self._build_prompt(
                question,
                visible_context,
                citation_metadata,
                target=target,
                fewshot_examples=fewshot_examples if target == "both" else None,
            )
            prompt = self._chat_text(user_prompt, system_prompt)
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=False,
            ).to(self.device)
            if encoded["input_ids"].shape[1] > max_input_tokens:
                raise ValueError(
                    f"Prompt vượt max_input_tokens: {encoded['input_ids'].shape[1]} > "
                    f"{max_input_tokens}"
                )
            attempt_seed = seed + attempt_index * 100_000
            self.torch.manual_seed(attempt_seed)
            if self.torch.cuda.is_available():
                self.torch.cuda.manual_seed_all(attempt_seed)
            if target == "both":
                max_new_tokens = int(self.config["max_new_tokens"])
            elif target == "lead":
                max_new_tokens = int(
                    self.config.get("targeted_lead_max_new_tokens", 96)
                )
            elif expanded_token_retry:
                max_new_tokens = int(
                    self.config.get(
                        "expanded_targeted_conclusion_max_new_tokens",
                        768,
                    )
                )
            else:
                max_new_tokens = int(
                    self.config.get("targeted_conclusion_max_new_tokens", 384)
                )
            initial_do_sample = self.decoding_mode in {"sample", "low_temperature"}
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": target == "both" and initial_do_sample,
                "repetition_penalty": 1.05,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if target == "both" and initial_do_sample:
                if self.decoding_mode == "low_temperature":
                    generation_kwargs.update(
                        {"temperature": 0.2, "top_p": 0.9, "top_k": 20}
                    )
                else:
                    generation_kwargs.update(
                        {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
                    )
            generated, elapsed = yield encoded, generation_kwargs
            text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            parsed = (
                parse_model_output(text)
                if target == "both"
                else parse_targeted_output(text, target)
            )
            attempts.append(
                {
                    "attempt": (
                        "initial"
                        if target == "both"
                        else "expanded_token_retry"
                        if expanded_token_retry
                        else "targeted_retry"
                    ),
                    "target": target,
                    "max_new_tokens": max_new_tokens,
                    "seed": attempt_seed,
                    "raw_output": text,
                    "parsed": parsed,
                    "latency_seconds": round(elapsed, 4),
                    "input_tokens": int(encoded["input_ids"].shape[1]),
                    "output_tokens": int(generated.shape[0]),
                }
            )

            if target == "both":
                missing_targets = []
                if not parsed.lead:
                    missing_targets.append("lead")
                if not parsed.conclusion:
                    missing_targets.append("conclusion")
                quality_issues = (
                    conclusion_quality_issues(question, visible_context, parsed.conclusion)
                    if self.enable_quality_retry and parsed.conclusion
                    else []
                )
                if quality_issues and "conclusion" not in missing_targets:
                    missing_targets.append("conclusion")
                attempts[0]["quality_issues"] = quality_issues
                targets.extend(missing_targets)
            elif (
                target == "conclusion"
                and not expanded_token_retry
                and should_expand_conclusion_retry(
                    raw_output=text,
                    parsed=parsed,
                    output_tokens=int(generated.shape[0]),
                    max_new_tokens=max_new_tokens,
                )
            ):
                targets.append("conclusion_expanded")

        initial = attempts[0]
        conclusion_retry_attempts = [
            attempt for attempt in attempts[1:] if attempt["target"] == "conclusion"
        ]
        initial_conclusion_retry = next(
            (
                attempt
                for attempt in conclusion_retry_attempts
                if attempt["attempt"] == "targeted_retry"
            ),
            None,
        )
        expanded_conclusion_retry = next(
            (
                attempt
                for attempt in conclusion_retry_attempts
                if attempt["attempt"] == "expanded_token_retry"
            ),
            None,
        )
        retry_attempts = {
            attempt["target"]: attempt
            for attempt in attempts[1:]
        }
        quality_retry_accepted = False
        quality_retry_selection_reasons: list[str] = []
        retry_conclusion = retry_attempts.get("conclusion", {}).get("parsed")
        if initial.get("quality_issues"):
            if retry_conclusion and retry_conclusion.valid:
                (
                    quality_retry_accepted,
                    quality_retry_selection_reasons,
                ) = should_accept_quality_retry(
                    question,
                    visible_context,
                    initial["parsed"].conclusion,
                    retry_conclusion.conclusion,
                )
            else:
                quality_retry_selection_reasons = ["retry_invalid"]
        soft_decision: dict[str, Any] = {
            "accepted": False,
            "initial_score": None,
            "retry_score": None,
            "margin": 0.75,
            "safety_reasons": [],
            "reasons": ["not_evaluated"],
            "initial_components": {},
            "retry_components": {},
        }
        if initial.get("quality_issues"):
            if retry_conclusion and retry_conclusion.valid:
                soft_decision = soft_retry_decision(
                    question,
                    visible_context,
                    initial["parsed"].conclusion,
                    retry_conclusion.conclusion,
                )
            else:
                soft_decision["reasons"] = ["retry_invalid"]
        resolved = resolve_generated_blocks(
            initial["parsed"],
            retry_attempts.get("lead", {}).get("parsed"),
            retry_conclusion,
            citation_metadata,
            prefer_retry_targets=(
                {"conclusion"} if quality_retry_accepted else set()
            ),
        )
        soft_resolved = resolve_generated_blocks(
            initial["parsed"],
            retry_attempts.get("lead", {}).get("parsed"),
            retry_conclusion,
            citation_metadata,
            prefer_retry_targets=(
                {"conclusion"} if soft_decision["accepted"] else set()
            ),
        )
        retry_targets = [attempt["target"] for attempt in attempts[1:]]
        retry_recovered = [
            attempt["target"]
            for attempt in attempts[1:]
            if attempt["parsed"].valid
        ]
        return {
            "raw_output": initial["raw_output"],
            "initial_prompt_protocol": self.prompt_mode,
            "decoding_mode": self.decoding_mode,
            "adapter_path": self.adapter_path,
            "fewshot_count": len(fewshot_examples or []),
            "base_prompt_max_input_tokens": base_prompt_max_input_tokens,
            "fewshot_examples": [
                {
                    "id": example["id"],
                    "question_type": example["question_type"],
                    "citation_group": example["citation_group"],
                    "selection_score": example["selection_score"],
                }
                for example in (fewshot_examples or [])
            ],
            "lead": resolved["lead"],
            "conclusion": resolved["conclusion"],
            "lead_source": resolved["lead_source"],
            "conclusion_source": resolved["conclusion_source"],
            "format_valid": resolved["model_blocks_complete"],
            "initial_format_valid": initial["parsed"].valid,
            "retry_used": bool(retry_targets),
            "quality_retry_used": bool(initial.get("quality_issues")),
            "quality_retry_reasons": initial.get("quality_issues", []),
            "quality_retry_accepted": quality_retry_accepted,
            "quality_retry_selection_reasons": quality_retry_selection_reasons,
            "soft_quality_retry_accepted": bool(soft_decision["accepted"]),
            "soft_quality_retry_decision": soft_decision,
            "soft_conclusion": soft_resolved["conclusion"],
            "soft_conclusion_source": soft_resolved["conclusion_source"],
            "soft_answer_complete": soft_resolved["answer_complete"],
            "token_limit_retry_used": bool(expanded_conclusion_retry),
            "token_limit_retry_recovered": bool(
                expanded_conclusion_retry
                and expanded_conclusion_retry["parsed"].valid
            ),
            "initial_retry_output_tokens": (
                initial_conclusion_retry["output_tokens"]
                if initial_conclusion_retry
                else None
            ),
            "expanded_retry_output_tokens": (
                expanded_conclusion_retry["output_tokens"]
                if expanded_conclusion_retry
                else None
            ),
            "expanded_retry_format_valid": (
                expanded_conclusion_retry["parsed"].valid
                if expanded_conclusion_retry
                else None
            ),
            "expanded_retry_accepted": bool(
                expanded_conclusion_retry
                and (
                    quality_retry_accepted
                    or (
                        not initial["parsed"].conclusion
                        and expanded_conclusion_retry["parsed"].valid
                    )
                )
            ),
            "expanded_retry_selection_reasons": (
                quality_retry_selection_reasons
                if expanded_conclusion_retry
                else []
            ),
            "retry_targets": retry_targets,
            "retry_recovered": retry_recovered,
            "retry_format_valid": (
                set(retry_targets) == set(retry_recovered) if retry_targets else None
            ),
            "model_blocks_complete": resolved["model_blocks_complete"],
            "answer_complete": resolved["answer_complete"],
            "parse_method": initial["parsed"].method,
            "attempts": [
                {
                    **{key: value for key, value in attempt.items() if key != "parsed"},
                    "parsed": {
                        "lead": attempt["parsed"].lead,
                        "conclusion": attempt["parsed"].conclusion,
                        "valid": attempt["parsed"].valid,
                        "method": attempt["parsed"].method,
                    },
                }
                for attempt in attempts
            ],
            "latency_seconds": round(
                sum(attempt["latency_seconds"] for attempt in attempts), 4
            ),
            "input_tokens": sum(attempt["input_tokens"] for attempt in attempts),
            "output_tokens": sum(attempt["output_tokens"] for attempt in attempts),
            "context_truncated": len(context_ids)
            < len(self.tokenizer(context, add_special_tokens=False)["input_ids"]),
        }


def make_prediction(row: dict, generation: dict) -> str:
    return assemble_answer(
        generation["lead"],
        row["oracle_context"]["text"],
        generation["conclusion"],
    )
