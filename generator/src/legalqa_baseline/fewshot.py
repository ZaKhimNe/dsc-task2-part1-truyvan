from __future__ import annotations

import re
from dataclasses import dataclass


WORD_RE = re.compile(r"[0-9a-zA-ZÀ-ỹĐđ]+", flags=re.UNICODE)
STOPWORDS = {
    "ai", "bao", "các", "có", "của", "được", "gì", "hay", "không", "là",
    "một", "nào", "những", "phải", "theo", "thế", "thì", "trong", "và", "về",
}


def question_type(question: str) -> str:
    text = " ".join(question.casefold().split())
    if re.search(r"\b(ai|đối tượng nào|chủ thể nào|người nào)\b", text):
        return "subject"
    if re.search(
        r"(?<!thẩm\s)\b(?:quyền(?:\s+lợi)?|nghĩa vụ|trách nhiệm)\b",
        text,
    ):
        return "rights_and_duties"
    if re.search(r"bao nhiêu|mức |thời hạn|thời gian|tỷ lệ|số lượng", text):
        return "number_or_time"
    if re.search(r"khi nào|trường hợp nào|điều kiện", text):
        return "condition"
    if re.search(
        r"trình tự|thủ tục|như thế nào|gồm những gì|bao gồm|"
        r"gồm những (?:nội dung|mục|bước|thành phần|loại) nào|"
        r"những (?:nội dung|vấn đề|giấy tờ|hồ sơ|tài liệu) (?:gì|nào)|"
        r"cần (?:chuẩn bị )?những (?:giấy tờ|hồ sơ|tài liệu) nào",
        text,
    ):
        return "procedure_or_list"
    if re.search(r"có được|được phép|có phải|hay không|không\?", text):
        return "yes_no"
    return "other"


def content_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in WORD_RE.findall(text)
        if len(token) > 1 and token.casefold() not in STOPWORDS
    }


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def is_usable_example(row: dict) -> bool:
    context = row.get("oracle_context", {}).get("text", "").strip()
    target = row.get("target", {})
    lead = target.get("lead", "").strip()
    conclusion = target.get("conclusion", "").strip()
    return bool(
        row.get("baseline_eligible")
        and context
        and lead
        and conclusion
        and len(context) <= 1_600
        and len(lead.split()) <= 60
        and len(conclusion.split()) <= 120
    )


@dataclass(frozen=True)
class SelectedExample:
    id: str
    question: str
    citation_group: str
    oracle_context: dict
    target: dict
    question_type: str
    selection_score: float

    def to_prompt_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "citation_group": self.citation_group,
            "oracle_context": self.oracle_context,
            "target": self.target,
            "question_type": self.question_type,
            "selection_score": self.selection_score,
        }


class FewShotSelector:
    """Chọn ví dụ train xác định, không dùng reference của mẫu đang chấm."""

    def __init__(self, train_rows: list[dict]):
        self.rows = [row for row in train_rows if is_usable_example(row)]
        if not self.rows:
            raise ValueError("Không có mẫu train hợp lệ cho few-shot")

    def select(self, query: dict, k: int = 2) -> list[dict]:
        query_type = question_type(query["question"])
        query_tokens = content_tokens(query["question"])
        query_group = query.get("citation_group", "")
        query_context_length = len(query["oracle_context"]["text"])
        ranked: list[tuple[float, str, dict]] = []
        for row in self.rows:
            if row["id"] == query.get("id"):
                continue
            if query_group and row.get("citation_group") == query_group:
                continue
            candidate_type = question_type(row["question"])
            lexical = jaccard(query_tokens, content_tokens(row["question"]))
            length_similarity = 1.0 - min(
                1.0,
                abs(len(row["oracle_context"]["text"]) - query_context_length)
                / max(1, query_context_length),
            )
            score = (
                10.0 * float(candidate_type == query_type)
                + 4.0 * lexical
                + length_similarity
            )
            ranked.append((score, str(row["id"]), row))
        ranked.sort(key=lambda item: (-item[0], item[1]))

        selected: list[dict] = []
        used_groups: set[str] = set()
        for score, _, row in ranked:
            group = row.get("citation_group", "")
            if group and group in used_groups:
                continue
            selected.append(
                SelectedExample(
                    id=str(row["id"]),
                    question=row["question"],
                    citation_group=group,
                    oracle_context=row["oracle_context"],
                    target=row["target"],
                    question_type=question_type(row["question"]),
                    selection_score=round(score, 6),
                ).to_prompt_dict()
            )
            if group:
                used_groups.add(group)
            if len(selected) == k:
                break
        if len(selected) < k:
            raise ValueError(f"Chỉ chọn được {len(selected)}/{k} ví dụ few-shot")
        return selected
