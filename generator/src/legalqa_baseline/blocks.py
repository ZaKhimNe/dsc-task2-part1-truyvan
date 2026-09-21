from __future__ import annotations

import re
from dataclasses import asdict, dataclass


CONCLUSION_MARKERS = (
    "Theo đó",
    "Như vậy",
    "Do đó",
    "Vì vậy",
    "Từ đó",
    "Đồng thời",
    "Như đã phân tích",
)

LEGAL_ID_RE = re.compile(
    r"\b\d{1,4}/\d{4}/(?:NĐ-CP|QH\d+|QĐ-[A-ZĐ]+|TT-[A-ZĐ]+|TTLT-[A-ZĐ-]+|"
    r"NQ-[A-ZĐ]+|CP|UBTVQH\d+|HĐTP)\b",
    flags=re.IGNORECASE,
)
ARTICLE_RE = re.compile(r"\bĐiều\s+\d+[a-zA-ZđĐ]?\b", flags=re.IGNORECASE)
CLAUSE_RE = re.compile(r"\bkhoản\s+\d+[a-zA-ZđĐ]?\b", flags=re.IGNORECASE)
POINT_RE = re.compile(r"\bđiểm\s+[a-zA-ZđĐ]\b", flags=re.IGNORECASE)
NAMED_DOCUMENT_RE = re.compile(
    r"\b(?:Bộ luật|Luật)\s+[^,:;\n]{2,100}?\s+(?:19|20)\d{2}\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class AnswerBlocks:
    lead: str
    quotation: str
    conclusion: str
    confidence: str
    method: str
    lead_start: int
    lead_end: int
    quotation_start: int
    quotation_end: int
    conclusion_start: int
    conclusion_end: int

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _first_nonempty_line_end(text: str) -> int:
    match = re.search(r"\n", text)
    return match.start() if match else len(text)


def _find_conclusion_start(text: str, minimum: int) -> tuple[int | None, str | None]:
    candidates: list[tuple[int, str]] = []
    marker_pattern = "|".join(re.escape(marker) for marker in CONCLUSION_MARKERS)
    for match in re.finditer(
        rf"(?im)^(?:\s*)(?P<marker>{marker_pattern})\b", text
    ):
        if match.start() >= minimum:
            candidates.append((match.start(), match.group("marker")))
    if not candidates:
        return None, None
    # Marker kết luận thường xuất hiện sau phần trích dẫn; chọn marker đầu tiên
    # sau lead để giữ toàn bộ chuỗi kết luận nhiều dòng.
    return min(candidates, key=lambda item: item[0])


def split_answer(answer: str) -> AnswerBlocks:
    text = normalize_newlines(answer)
    if not text:
        return AnswerBlocks("", "", "", "low", "empty", 0, 0, 0, 0, 0, 0)

    lead_end = _first_nonempty_line_end(text)
    lead = text[:lead_end].strip()
    quote_start = lead_end
    while quote_start < len(text) and text[quote_start] in "\n \t":
        quote_start += 1

    conclusion_start, marker = _find_conclusion_start(text, quote_start + 1)
    if conclusion_start is not None:
        quotation = text[quote_start:conclusion_start].strip()
        conclusion = text[conclusion_start:].strip()
        confidence = "high" if lead and quotation and conclusion else "low"
        method = f"first_line+marker:{marker}"
    else:
        # Không suy đoán đoạn cuối là conclusion: giữ toàn bộ phần sau lead làm
        # quotation và đánh dấu low để mẫu không được dùng cho bake-off mặc định.
        quotation = text[quote_start:].strip()
        conclusion = ""
        conclusion_start = len(text)
        confidence = "low"
        method = "first_line+no_marker"

    quotation_end = conclusion_start
    return AnswerBlocks(
        lead=lead,
        quotation=quotation,
        conclusion=conclusion,
        confidence=confidence,
        method=method,
        lead_start=0,
        lead_end=lead_end,
        quotation_start=quote_start,
        quotation_end=quotation_end,
        conclusion_start=conclusion_start,
        conclusion_end=len(text),
    )


def citation_signature(answer: str, sample_id: str) -> str:
    first_line = normalize_newlines(answer).split("\n", 1)[0]
    laws = sorted({item.upper() for item in LEGAL_ID_RE.findall(first_line)})
    articles = sorted({item.lower() for item in ARTICLE_RE.findall(first_line)})
    clauses = sorted({item.lower() for item in CLAUSE_RE.findall(first_line)})
    if not laws and not articles:
        return f"ungrouped:{sample_id}"
    return "|".join(laws + articles + clauses)


def extract_citation_metadata(lead: str) -> dict[str, list[str]]:
    """Chỉ lấy dữ kiện căn cứ, không đưa nguyên lead vào input model."""

    def unique(pattern: re.Pattern[str]) -> list[str]:
        seen: set[str] = set()
        values: list[str] = []
        for value in pattern.findall(lead):
            normalized = re.sub(r"\s+", " ", value).strip()
            key = normalized.casefold()
            if key not in seen:
                seen.add(key)
                values.append(normalized)
        return values

    return {
        "document_numbers": unique(LEGAL_ID_RE),
        "named_documents": unique(NAMED_DOCUMENT_RE),
        "articles": unique(ARTICLE_RE),
        "clauses": unique(CLAUSE_RE),
        "points": unique(POINT_RE),
    }


def assemble_answer(lead: str, quotation: str, conclusion: str) -> str:
    return "\n".join(part.strip() for part in (lead, quotation, conclusion) if part.strip())


def baseline_eligibility(blocks: AnswerBlocks) -> tuple[bool, list[str]]:
    """Lọc tập bake-off về đúng giả định ba khối đơn.

    Dữ liệu gốc có một số answer lặp nhiều vòng căn cứ/kết luận hoặc dính tiêu
    đề trang web. Những mẫu đó vẫn được lưu, nhưng không dùng để so model ở
    vòng baseline hybrid đầu tiên.
    """

    reasons: list[str] = []
    if blocks.confidence != "high":
        reasons.append("split_not_high_confidence")
    if len(blocks.lead) > 600:
        reasons.append("lead_too_long")
    if len(blocks.quotation) > 12_000:
        reasons.append("quotation_too_long")
    if len(blocks.conclusion) > 1_800:
        reasons.append("conclusion_too_long")
    if re.search(
        r"(?im)^(?:Căn cứ\b|Trước đây,\s*căn cứ\b|Hiện nay,\s*theo\s+quy\s+định\b|"
        r"Theo\s+(?:Điều|khoản)\b|Tại\s+Điều\b)",
        blocks.conclusion,
    ):
        reasons.append("nested_legal_quote_in_conclusion")
    if re.search(r"(?i)\(Hình từ Internet\)|Thư Viện Pháp Luật", blocks.conclusion):
        reasons.append("web_page_noise")
    return not reasons, reasons
