"""
B0 — Gán nhãn tự động (Auto-label Alignment).

Bài toán: text reuse detection / source attribution — đã có `answer` hoàn
chỉnh, tìm ngược lại nó trích từ Khoản/Dieu nào trong corpus. KHÁC retrieval
(retrieval là chưa biết đáp án, đi tìm; B0 là đã biết đáp án, tìm nguồn).

Input: question + answer (từ train.json/warmup.json) + B2 index đã build
(`retrieve.build_index`) + `parsed_corpus` (`io_utils.load_parsed_corpus()`).

Output: {question_id: [MatchedSpan, ...]} — xem class MatchedSpan bên dưới.
Có thể nhiều MatchedSpan/câu hỏi nếu answer trích từ ≥2 văn bản (12.2% ca).

REBUILD:
  Bản cũ tự viết 1 global shingle n-gram index riêng cho toàn corpus để tìm
  candidate — OOM ngay cả sau tối ưu (băm key ra int), chưa từng chạy đúng
  nghĩa full corpus (chỉ chạy được bằng pool ngẫu nhiên 1,000 văn bản, xem
  docs/EXPERIMENT_LOG.md entry [B3] 06/08). Bản mới BỎ HẲN cơ chế đó, dùng
  lại `retrieve.search_units()` (B2 Layer 1 BM25 cấp văn bản + Expand sang
  Khoản) để tìm candidate — B2 đã có giải pháp OOM riêng (MAX_TOKENIZE_CHARS),
  không viết 2 lần logic retrieval.

  Hệ quả: candidate giờ là UnitCandidate (Khoản/Dieu, ngắn — vài trăm ký tự)
  thay vì cả văn bản (có thể dài hàng chục nghìn ký tự) — không cần cơ chế
  "tìm anchor n-gram rồi cắt cửa sổ alignment" của bản cũ nữa (đã bỏ hẳn
  _rolling_ngram_hashes/_locate_anchor_positions/_find_best_anchor_cluster).
  align_span giờ chạy Smith-Waterman FULL trên toàn unit, chỉ cap độ dài cho
  case unit_type="doc_fallback" (có thể vẫn là cả văn bản dài nếu B1 không
  tách được cấu trúc) bằng MAX_ALIGN_CHARS — cùng tinh thần B2's
  MAX_TOKENIZE_CHARS, tránh lặp lại đúng lỗi OOM cũ.

Workflow — 3 bước chuẩn học thuật cho text reuse detection:
  0. _extract_quote_core: trích riêng Khối 2 (giữa "như sau:" và "Theo
     đó/Như vậy/Tóm lại") trước khi tìm/align — đo cải thiện +0.519 điểm
     align (xem docs/EXPERIMENT_LOG.md 06/08). Answer/quote < MIN_QUOTE_WORDS
     (5 từ) sau bước này -> bỏ qua, không gán nhãn (tránh false positive từ
     cụm quá ngắn, phổ biến, dễ match nhầm nhiều nơi).
  1. retrieve.search_units(): B2 Layer 1 (BM25 cấp văn bản, multi-query) +
     Expand (mở rộng sang Khoản/Dieu trong top-N văn bản) -> list
     UnitCandidate.
  2. align_span: Smith-Waterman word-level FULL trên từng UnitCandidate (đủ
     nhanh vì unit ngắn) -> chọn unit có similarity cao nhất.
  3. Similarity score (chuẩn hoá theo điểm SW lý thuyết tối đa) = đồng thời
     là thước đo "recall trần" — unit-level score thấp dù đã tìm đúng văn
     bản nghĩa là B1 chunk làm vỡ mất câu trả lời (vắt qua ranh giới Khoản).

Chuẩn hoá Unicode NFC áp cho cả answer và unit text trước khi tokenize/align
(rủi ro chưa từng xử lý ở bản cũ: crawl web và answer do người gõ có thể ở 2
dạng Unicode khác nhau nhìn giống nhau nhưng so sánh chuỗi thất bại âm thầm).

Edge case đã quyết (dựa số liệu thật warmup.json, xem docs/EXPERIMENT_LOG.md):
  - Đa văn bản (12.2%): answer tách theo marker "Trước đây..." (nếu có)
    thành các phần, mỗi phần tìm 1 candidate tốt nhất. GIỚI HẠN V1 (để sau,
    đã xác nhận 12/08): case đa văn bản KHÔNG có marker (2 trích dẫn liên
    tiếp trong 1 đoạn không sửa đổi) hiện chỉ ra 1 MatchedSpan, chưa tách.
  - Có sửa đổi/bổ sung (13.2%): detect_role gắn "hiện hành"/"trước đây".
  - Ngưỡng confidence (B0_CONFIDENCE_DROP/TRUST): khởi điểm 0.3/0.6, SẼ
    hiệu chỉnh lại sau khi đo phân phối score thật trên full warmup.json.

TRẠNG THÁI: rebuild 12/08/2026, đang test trên data thật.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Literal, TypedDict

from src.common import config
from src.common.io_utils import QAItem
from src.b2_retrieval.retrieve import BM25Index, UnitCandidate, search_units

DEFAULT_SHINGLE_N = 6  # vẫn dùng cho no-op tương thích cũ nếu cần, KHÔNG dùng trong candidate retrieval nữa
MIN_QUOTE_WORDS = 5  # answer/quote ngắn hơn ngần này (sau _extract_quote_core) -> bỏ qua, không gán nhãn
MAX_ALIGN_CHARS = 50_000  # cap cho unit_type="doc_fallback" (có thể là cả văn bản dài) — tránh OOM khi tokenize

_TOKEN_RE = re.compile(r"\S+")
_AMEND_MARKER_RE = re.compile(r"Trước đây,?\s*(căn cứ|theo)?.*?quy định như sau:?", re.IGNORECASE)


class MatchedSpan(TypedDict):
    context_id: str
    khoan_id: str | None
    unit_type: Literal["khoan", "dieu_fallback", "doc_fallback"]
    matched_span: str
    confidence: float
    role: Literal["hiện hành", "trước đây"] | None


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Tách text theo khoảng trắng, GIỮ vị trí ký tự (start, end) để sau này
    cắt lại đúng nguyên văn từ passage gốc."""
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def align_span(answer: str, candidate_text: str, max_align_chars: int = MAX_ALIGN_CHARS) -> tuple[str, float]:
    """Smith-Waterman word-level FULL trên `candidate_text` (không cắt cửa
    sổ theo anchor n-gram như bản cũ — không cần nữa vì candidate giờ là 1
    unit Khoản/Dieu ngắn, không phải cả văn bản). `candidate_text` bị cắt
    tới `max_align_chars` nếu dài hơn (case unit_type="doc_fallback" vẫn có
    thể là cả văn bản chưa tách được cấu trúc, có thể dài hàng triệu ký tự).

    Trả về (matched_span_text, similarity_score trong [0, 1])."""
    candidate_text = _nfc(candidate_text)[:max_align_chars]
    answer = _nfc(answer)

    passage_spans = _tokenize_with_spans(candidate_text)
    passage_tokens = [tok for tok, _, _ in passage_spans]
    answer_tokens = answer.split()

    n_a, n_p = len(answer_tokens), len(passage_tokens)
    if n_a == 0 or n_p == 0:
        return "", 0.0

    MATCH, MISMATCH, GAP = 2, -1, -1
    H = [[0] * (n_p + 1) for _ in range(n_a + 1)]
    max_score = 0
    max_pos = (0, 0)
    for i in range(1, n_a + 1):
        ai = answer_tokens[i - 1]
        row, prev_row = H[i], H[i - 1]
        for j in range(1, n_p + 1):
            diag = prev_row[j - 1] + (MATCH if ai == passage_tokens[j - 1] else MISMATCH)
            up = prev_row[j] + GAP
            left = row[j - 1] + GAP
            cell = max(0, diag, up, left)
            row[j] = cell
            if cell > max_score:
                max_score = cell
                max_pos = (i, j)

    if max_score == 0:
        return "", 0.0

    i, j = max_pos
    end_j = j
    while i > 0 and j > 0 and H[i][j] > 0:
        ai = answer_tokens[i - 1]
        pj = passage_tokens[j - 1]
        diag = H[i - 1][j - 1] + (MATCH if ai == pj else MISMATCH)
        if H[i][j] == diag:
            i -= 1
            j -= 1
        elif H[i][j] == H[i - 1][j] + GAP:
            i -= 1
        else:
            j -= 1
    start_j = j

    if start_j >= end_j:
        return "", 0.0

    char_start = passage_spans[start_j][1]
    char_end = passage_spans[end_j - 1][2]
    matched_text = candidate_text[char_start:char_end]

    theoretical_max = MATCH * n_a
    similarity = max(0.0, min(1.0, max_score / theoretical_max)) if theoretical_max else 0.0
    return matched_text, similarity


def detect_role(answer_segment: str) -> Literal["hiện hành", "trước đây"] | None:
    """Nhận diện pattern 'Trước đây, căn cứ ... quy định như sau:' ở ĐẦU
    answer_segment để gắn role "trước đây". None nếu không thấy."""
    if _AMEND_MARKER_RE.match(answer_segment.strip()):
        return "trước đây"
    return None


_QUOTE_START_RE = re.compile(r"như sau:?\s*", re.IGNORECASE)
_QUOTE_END_RE = re.compile(r"(Theo đó|Như vậy|Tóm lại)\b", re.IGNORECASE)


def _extract_quote_core(text: str) -> str:
    """Trích phần Khối 2 (trích nguyên văn điều luật) nằm giữa 'như sau:' và
    'Theo đó/Như vậy/Tóm lại'. Fallback: không tìm được đủ 2 marker thì trả
    nguyên text."""
    start_m = _QUOTE_START_RE.search(text)
    if not start_m:
        return text
    start = start_m.end()
    end_m = _QUOTE_END_RE.search(text, pos=start)
    end = end_m.start() if end_m else len(text)
    core = text[start:end].strip()
    return core if core else text


def _split_by_amend_marker(answer: str) -> list[tuple[str, Literal["hiện hành", "trước đây"] | None]]:
    """Tách answer thành các phần theo marker sửa đổi/bổ sung. Không có
    marker -> trả về nguyên answer, role=None (đa số trường hợp)."""
    m = _AMEND_MARKER_RE.search(answer)
    if not m:
        return [(answer, None)]
    hien_hanh = answer[: m.start()].strip()
    truoc_day = answer[m.start() :].strip()
    parts: list[tuple[str, Literal["hiện hành", "trước đây"] | None]] = []
    if hien_hanh:
        parts.append((hien_hanh, "hiện hành"))
    if truoc_day:
        parts.append((truoc_day, "trước đây"))
    return parts or [(answer, None)]


def _prefilter_units(
    query_text: str, units: list[UnitCandidate], top_n: int, already_ranked: bool = False
) -> list[UnitCandidate]:
    """Lọc rẻ TRƯỚC khi chạy Smith-Waterman (đắt) — bắt buộc phải có bước
    này, không phải tối ưu sớm: `search_units(top_k_docs=20)` trả về
    2,000-3,000 unit/câu hỏi (đo thật, xem docs/EXPERIMENT_LOG.md entry
    [B0] 12/08) — chạy SW full trên TẤT CẢ unit đó cho 500 câu ước tính mất
    HÀNG GIỜ (đã phải kill tiến trình thật đang chạy vì quá chậm).

    `already_ranked=True` (units đã qua Layer 1.5 — BM25 cấp Khoản, xem
    retrieve.rerank_units_khoan): TIN thứ hạng đó, chỉ cắt top_n, KHÔNG tự
    tính lại — BM25 (có IDF + length norm) là tín hiệu tốt hơn overlap thô
    dưới đây. Bug đã tự phát hiện trước khi chạy thật (13/08): nếu không có
    cờ này, hàm luôn tự tính lại bằng overlap token thô bất kể input đã
    được rerank ngoài hay chưa — làm Layer 1.5 hoàn toàn VÔ TÁC DỤNG với B0
    (bị ghi đè ngay lập tức), lãng phí công build/wire mà không có gì thay
    đổi.

    `already_ranked=False` (mặc định, tương thích ngược): overlap token thô
    (giống B4 select_khoan) — O(n) rẻ, dùng khi không có Layer 1.5."""
    if len(units) <= top_n:
        return units
    if already_ranked:
        return units[:top_n]
    query_tokens = set(query_text.lower().split())
    scored = [(len(query_tokens & set(u["text"].lower().split())), u) for u in units]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [u for _, u in scored[:top_n]]


def _best_unit_match(
    query_text: str, units: list[UnitCandidate], prefilter_top_n: int = 20, already_ranked: bool = False
) -> tuple[UnitCandidate, str, float] | None:
    candidates = _prefilter_units(query_text, units, prefilter_top_n, already_ranked)
    best: tuple[UnitCandidate, str, float] | None = None
    for unit in candidates:
        span, score = align_span(query_text, unit["text"])
        if best is None or score > best[2]:
            best = (unit, span, score)
    return best


def label_answer(
    qa_item: QAItem,
    bm25_index: BM25Index,
    parsed_corpus: dict[str, dict],
    top_k_docs: int = 20,
    khoan_index: BM25Index | None = None,
    khoan_position_index: dict[str, int] | None = None,
) -> list[MatchedSpan]:
    """Gán nhãn cho 1 câu hỏi: mỗi phần answer (sau tách marker sửa đổi) ->
    search_units (B2, có Layer 1.5 nếu truyền khoan_index) tìm candidate ->
    align_span chọn unit tốt nhất.

    `khoan_index`/`khoan_position_index`: để None (mặc định) -> B0 chạy như
    cũ (Layer 1 + overlap prefilter thô). Truyền vào -> bật Layer 1.5 (BM25
    cấp Khoản rerank candidate trước khi prefilter, TIN thứ hạng đó thay vì
    tính lại bằng overlap thô — xem `_prefilter_units`)."""
    answer = qa_item.get("answer")
    if not answer:
        return []

    use_layer15 = khoan_index is not None and khoan_position_index is not None
    matches: list[MatchedSpan] = []
    for part_text, role in _split_by_amend_marker(answer):
        query_text = _extract_quote_core(part_text)
        if len(query_text.split()) < MIN_QUOTE_WORDS:
            continue
        units = search_units(
            bm25_index,
            query_text,
            parsed_corpus,
            top_k_docs=top_k_docs,
            khoan_index=khoan_index,
            khoan_position_index=khoan_position_index,
        )
        if not units:
            continue
        best = _best_unit_match(query_text, units, already_ranked=use_layer15)
        if best is None:
            continue
        unit, span, score = best
        if score >= config.B0_CONFIDENCE_DROP:
            matches.append(
                MatchedSpan(
                    context_id=unit["context_id"],
                    khoan_id=unit["unit_id"] if unit["unit_type"] == "khoan" else None,
                    unit_type=unit["unit_type"],
                    matched_span=span,
                    confidence=score,
                    role=role,
                )
            )
    return matches


def label_dataset(
    qa_set: dict[str, QAItem],
    bm25_index: BM25Index,
    parsed_corpus: dict[str, dict],
    top_k_docs: int = 20,
    khoan_index: BM25Index | None = None,
    khoan_position_index: dict[str, int] | None = None,
) -> dict[str, list[MatchedSpan]]:
    """Chạy B0 trên toàn bộ 1 tập (train/warmup). Output {question_id:
    [MatchedSpan,...]} — dùng để ghi ra outputs/b0_labels_<tên tập>.json."""
    return {
        qid: label_answer(item, bm25_index, parsed_corpus, top_k_docs, khoan_index, khoan_position_index)
        for qid, item in qa_set.items()
    }
