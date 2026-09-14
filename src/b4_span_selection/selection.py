"""
B4 — Chọn đoạn trích cụ thể (span selection).

Bài toán: extractive selection nhẹ — từ 1 Khoản ứng viên (đã có nhờ B1 chunk
+ B2 retrieve) và câu hỏi, chọn ra đơn vị con (Điểm "a)/b)/..." hoặc câu)
liên quan nhất, thay vì trả nguyên cả Khoản (có thể dài, chứa nhiều điểm
không liên quan tới câu hỏi cụ thể — ví dụ thật warmup.json id 97213: Thông
tư 64/2013/TT-BGTVT có 6 khoản, câu hỏi chỉ hỏi về đường bộ, chỉ Khoản 5
đúng, 5 khoản còn lại nhiễu dù cùng văn bản).

KHÁC B0: B0 làm việc NGƯỢC (có answer, tìm nguồn) ở dev-time để tạo nhãn.
B4 làm việc XUÔI (có câu hỏi, CHƯA có answer) ở serving-time thật — không
thể tái dùng thẳng thuật toán marker-based của B0 (`_extract_quote_core`)
vì lúc serving không có answer để tách marker "như sau:"/"Theo đó". Nhãn
B0 sinh ra (`MatchedSpan.matched_span`) chỉ dùng để ĐÁNH GIÁ B4 (so khớp
xem B4 chọn đúng không), không phải input của B4.

Input: question (str) + khoan_text (str — text của 1 Khoản ứng viên, lấy từ
`Dieu["khoan"][i]["text"]` sau B1, hoặc từ kết quả B2 nếu đã index ở
granularity Khoản).

Output: str — đoạn trích con được chọn trong khoan_text, NGUYÊN VĂN (không
paraphrase, không thêm/bớt từ).

Thuật toán (baseline v0, 0 tham số học — nhất quán với B2 và ưu tiên ngân
sách tham số toàn hệ thống):
  1. Tách khoan_text thành các đơn vị theo ranh giới Điểm tự nhiên đã có sẵn
     trong text ("\\n\\n" — B1 giữ lại ranh giới đoạn thật khi gộp Điểm vào
     Khoản, xem b1_parser/parser.py). Khoản không chia Điểm (chỉ 1 đoạn) ->
     tách fallback theo câu/mệnh đề (kết thúc "."/";").
  2. Tokenize word-level đơn giản (.split(), không dùng underthesea — bước
     nhẹ, không cần word-segmentation chính xác cho việc đếm overlap thô).
  3. Chấm điểm overlap = |token chung ∩ giữa unit và câu hỏi| (tập hợp,
     không trọng số — baseline v0, có thể nâng cấp lên BM25/TF-IDF sau khi
     đo thấy cần).
  4. Trả về unit điểm cao nhất. Mọi unit đều 0 điểm (không overlap từ nào)
     -> fallback trả NGUYÊN khoan_text (an toàn hơn đoán bừa 1 unit sai).

Ngoài `select_span` (chọn ĐƠN VỊ CON trong 1 Khoản), module còn có
`select_khoan` (chọn KHOẢN NÀO trong nhiều Khoản ứng viên của cùng 1 văn
bản — đúng tình huống ví dụ warmup id 97213 ở trên: B2 hiện trả về cả văn
bản, chưa index ở granularity Khoản, nên cần bước này để thu hẹp từ "cả văn
bản" xuống "đúng Khoản" trước khi gọi `select_span` thu hẹp tiếp xuống
"đúng đơn vị con"). Cùng thuật toán overlap token thô, khác cấp độ áp dụng.

TRẠNG THÁI: ĐÃ CODE (06/08/2026), test bằng cách đối chiếu với nhãn B0
(`MatchedSpan.matched_span` — không phải ground truth hoàn hảo cho B4 vì B0
định vị VĂN BẢN nguồn, B4 cần định vị ĐÚNG ĐIỂM/CÂU, nhưng đủ để kiểm tra
sơ bộ B4 có chọn đúng vùng lân cận không) — xem docs/EXPERIMENT_LOG.md entry
[B4] 06/08 và experiments/demo_b4_span_selection.py. `select_khoan` test
qua trong `experiments/demo_b_end_to_end.py` (chưa có test riêng biệt).
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.b1_parser.parser import Khoan

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\n+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+")


def _split_units(khoan_text: str) -> list[str]:
    """Tách khoan_text thành các đơn vị con để chấm điểm. Ưu tiên ranh giới
    Điểm ("\\n\\n", do B1 giữ lại) — nếu chỉ có 1 unit, tách tiếp theo
    câu/mệnh đề (kết thúc bằng "." hoặc ";")."""
    units = [u.strip() for u in _PARAGRAPH_SPLIT_RE.split(khoan_text) if u.strip()]
    if len(units) <= 1:
        units = [u.strip() for u in _SENTENCE_SPLIT_RE.split(khoan_text) if u.strip()]
    if not units:
        return [khoan_text.strip()] if khoan_text.strip() else []
    return units


def select_span(question: str, khoan_text: str) -> str:
    """⚠️ ĐO ĐƯỢC LÀ CÓ HẠI — KHÔNG DÙNG trong pipeline chính (17/08/2026).

    Đo trên 151 ca mà Khoản đúng CÓ trong tập candidate (nhãn B0 cấp Khoản,
    `qa_train` dev sample): hàm này chỉ giữ được đoạn đáp án gold ở
    **67/151 = 44.4%** — nghĩa là **hơn NỬA số lần nó cắt bỏ đúng phần chứa
    câu trả lời**. Kết hợp: (a) nguyên tắc đã chốt với người dùng "ưu tiên
    recall, thà dư hơn thiếu", (b) tài liệu nhóm ghi METEOR phạt viết THIẾU
    nặng gấp 3-4 lần viết DƯ → cắt nhỏ Khoản là đánh đổi SAI HƯỚNG.
    Xem docs/EXPERIMENT_LOG.md entry [B2/B4] 17/08.

    GIỮ hàm lại (không xoá) để: tái đo nếu sau này có generator bị giới hạn
    context chặt, hoặc nếu tìm được thuật toán chọn span tốt hơn overlap
    token thô. Pipeline hiện tại trả NGUYÊN Khoản.

    Chọn đơn vị con liên quan nhất tới question trong khoan_text.

    Rỗng input -> "". Chỉ 1 unit (không tách được) -> trả nguyên khoan_text.
    Không unit nào overlap với câu hỏi -> fallback trả nguyên khoan_text
    (an toàn hơn đoán bừa khi không có tín hiệu)."""
    if not khoan_text or not khoan_text.strip():
        return ""

    units = _split_units(khoan_text)
    if len(units) <= 1:
        return khoan_text.strip()

    question_tokens = set(question.lower().split())
    best_unit = units[0]
    best_score = -1
    for unit in units:
        unit_tokens = set(unit.lower().split())
        score = len(question_tokens & unit_tokens)
        if score > best_score:
            best_score = score
            best_unit = unit

    if best_score <= 0:
        return khoan_text.strip()
    return best_unit


def select_top_n_khoan(
    question: str, khoan_list: list["Khoan"], top_n: int = 3, already_ranked: bool = False
) -> list["Khoan"]:
    """Chọn TOP-N Khoản liên quan nhất (thay vì chỉ 1 — xem `select_khoan`
    bên dưới) — dùng khi B6 cần trả NHIỀU context cho Generator thay vì 1,
    theo nguyên tắc ưu tiên recall đã chốt với người dùng 13/08 (thà dư hơn
    thiếu) — tăng khả năng không bỏ sót nguồn đúng.

    `already_ranked=True` (khoan_list đã qua Layer 1.5/Layer 2 của B2, mỗi
    phần tử có field "score" đáng tin) -> TIN thứ hạng đó, chỉ cắt top_n,
    KHÔNG tự tính lại (nhất quán với `_prefilter_units` của B0 — BM25/dense
    là tín hiệu tốt hơn overlap thô dưới đây). `already_ranked=False` (mặc
    định, tương thích ngược) -> tự tính overlap token thô như `select_khoan`.

    `khoan_list` rỗng -> []."""
    if not khoan_list:
        return []
    if already_ranked:
        return khoan_list[:top_n]

    question_tokens = set(question.lower().split())
    scored = [(len(question_tokens & set(k["text"].lower().split())), k) for k in khoan_list]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [k for _, k in scored[:top_n]]


def select_khoan(question: str, khoan_list: list["Khoan"]) -> "Khoan | None":
    """Chọn Khoản liên quan nhất tới question trong số nhiều Khoản ứng viên
    (cùng 1 văn bản/Điều — dùng khi B2 mới thu hẹp tới cấp văn bản, chưa
    tới cấp Khoản). Cùng thuật toán overlap token thô với `select_span`,
    áp dụng ở cấp Khoản thay vì cấp đơn vị con.

    `khoan_list` rỗng -> None. Không Khoản nào overlap câu hỏi -> trả Khoản
    ĐẦU TIÊN (an toàn hơn None khi caller cần luôn có 1 kết quả để đóng gói,
    nhất quán với fallback "không đoán bừa" của `select_span` ở việc KHÔNG
    tự tin chọn nhầm — chỉ khác vì ở đây phải trả 1 Khoản cụ thể, không có
    lựa chọn "trả nguyên văn bản" như `select_span` có "trả nguyên khoan_text")."""
    if not khoan_list:
        return None

    question_tokens = set(question.lower().split())
    best_khoan = khoan_list[0]
    best_score = -1
    for khoan in khoan_list:
        khoan_tokens = set(khoan["text"].lower().split())
        score = len(question_tokens & khoan_tokens)
        if score > best_score:
            best_score = score
            best_khoan = khoan
    return best_khoan
