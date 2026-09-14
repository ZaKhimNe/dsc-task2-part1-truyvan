"""
B1 — Parser Điều/Khoản (Phase A).

Đã xác nhận (06/08/2026): nhóm IR (Task 1) và nhóm QA (Task 2, nhóm của B)
là 2 bài toán tách biệt hoàn toàn, không liên quan/không cần chia sẻ hạ
tầng — khác với suy đoán trong `B_technical_handoff.md` mục 1. Schema dưới
đây do B tự quyết định hoàn toàn, không còn phụ thuộc bên ngoài nào.

Bài toán: tách mỗi văn bản luật (passage trong context_*.json) thành cây
Điều -> Khoản -> (Điểm nếu có), để B0 định vị span chính xác và B2 index ở
đúng granularity.

Input: passage: str (1 văn bản, đã qua io_utils.clean_passage).

Output: list[Dieu] — xem class Dieu/Khoan bên dưới.

Workflow: regex/rule-based match pattern "Điều X." / "Khoản Y." sau khi
chuẩn hoá whitespace (\\r\\n xen \\n\\n do crawl — xem docs/DATA_NOTES.md
mục 3). Fallback bắt buộc: nếu không match cấu trúc chuẩn (TCVN/QCVN, văn
bản Đảng, ~15% corpus không nhận diện được loại — xem DATA_NOTES.md mục 3),
trả về toàn bộ passage như 1 Dieu duy nhất, không crash.

Case "sửa đổi, bổ sung" (13.2% câu hỏi liên quan — đã chốt 06/08/2026): B1
KHÔNG xử lý, để B0 nhận diện qua pattern "Trước đây, căn cứ..." (đã chốt sẵn
trong b0_autolabel/label.py) — tránh trùng logic ở 2 chỗ.

API: `parse_document(context_id, passage) -> ParsedDoc`. Mỗi Dieu/Khoan có
`dieu_id`/`khoan_id` duy nhất trong toàn corpus (`{context_id}_{dieu_so}`,
`{context_id}_{dieu_so}_{khoan_so}`) để B2/B4 tham chiếu qua lại không lệch.

Đuôi hành chính "Nơi nhận:" / "PHỤ LỤC..." (mẫu đơn, bảng biểu, danh mục đi
kèm — không phải nội dung Điều/Khoản) bị cắt riêng vào `phu_luc_raw` trước
khi tách Điều (xem `_split_tail`). Không cắt sẽ dính vào Điều cuối và các
dòng đánh số trong bảng bị bắt nhầm thành Khoản giả (đo được thật: "Khoản
416" từ số hiệu tài khoản kế toán — xem docs/EXPERIMENT_LOG.md).

Unit test mẫu: context_740.json = Quyết định 5868/QĐ-BYT — dùng file này
test trước; context_166280, context_231867 — case có Phụ lục cần verify
không còn Khoản giả.
"""
from __future__ import annotations

import re
from typing import TypedDict


class Khoan(TypedDict):
    khoan_id: str
    khoan_so: str
    text: str
    char_start: int  # offset ký tự trong main_text (normalized, sau _split_tail) — xem _strip_with_offset
    char_end: int


class Dieu(TypedDict):
    dieu_id: str
    dieu_so: str
    dieu_tieu_de: str
    text: str
    char_start: int  # offset ký tự trong main_text — dùng để đo quote (B0) có vắt qua ranh giới Khoan/Dieu không
    char_end: int
    khoan: list[Khoan]


class ParsedDoc(TypedDict):
    parse_status: str  # "matched" | "fallback"
    dieu: list[Dieu]
    phu_luc_raw: str | None


# "\r\n" = gãy dòng GIỮA CÂU do crawl bẻ dòng dài (xem docs/DATA_NOTES.md mục
# 3) -> phải nối bằng dấu cách, KHÔNG phải "\n" (nếu coi là "\n" thì regex
# "^..." theo dòng sẽ nhầm từ giữa câu thành đầu dòng mới -> cắt cụt tiêu đề
# hoặc tệ hơn, coi nhầm 1 từ/số giữa câu là header Điều/Khoản giả — bug thật
# đã đo được: "Điều 3. Cơ cấu tổ chức và hoạt\r\n\nđộng" bị cắt tiêu đề thành
# "...và hoạt" nếu gộp \r\n = \n). "\n\n" (không có \r ngay trước) mới là
# ranh giới đoạn/cấu trúc thật -> giữ nguyên làm điểm neo cho "^".
_MID_LINE_BREAK_RE = re.compile(r"\r\n\n?")  # "\r\n" đo được luôn kèm thêm 1 "\n" thừa ngay sau trong data thật
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

# "Điều 12. Tiêu đề" hoặc "Điều 12: Tiêu đề" hoặc "Điều 12" (không tiêu đề) ở đầu dòng.
_DIEU_RE = re.compile(
    r"^Điều\s+(?P<so>\d+[a-zA-Z]?)\s*[.:]?\s*(?P<tieu_de>[^\n]*)$",
    re.MULTILINE,
)

# "1. " / "1) " ở đầu dòng — Khoản trong phạm vi 1 Điều.
_KHOAN_RE = re.compile(r"^(?P<so>\d+)[.)]\s*", re.MULTILINE)

# Đuôi hành chính đứng riêng 1 dòng: "Nơi nhận:" (danh sách người nhận) hoặc
# "PHỤ LỤC" (mẫu đơn/bảng biểu/danh mục kèm theo). Không phải nội dung
# Điều/Khoản — nếu không cắt trước khi tách Khoản, các dòng đánh số bên
# trong (số thứ tự bảng, gạch đầu dòng...) bị _KHOAN_RE bắt nhầm thành
# Khoản giả (đo được thật: "Khoản 416" từ số hiệu tài khoản kế toán trong
# Phụ lục — xem docs/EXPERIMENT_LOG.md). Dòng ngắn (<=20 ký tự sau marker)
# để tránh cắt nhầm khi "Phụ lục" chỉ được nhắc giữa câu (vd "...theo Phụ lục
# I ban hành kèm theo...").
_TAIL_MARKER_RE = re.compile(
    r"^\s*(?:Nơi nhận\s*:|PHỤ\s+LỤC(?:\s+[IVXLCM\d]+)?\s*)$",
    re.MULTILINE | re.IGNORECASE,
)


def _split_tail(normalized: str) -> tuple[str, str | None]:
    """Cắt đuôi hành chính (Nơi nhận/Phụ lục) khỏi phần nội dung chính.

    Trả về (main_text, phu_luc_raw). phu_luc_raw = None nếu không tìm thấy
    marker nào (văn bản không có đuôi này)."""
    m = _TAIL_MARKER_RE.search(normalized)
    if m is None:
        return normalized, None
    return normalized[: m.start()].strip(), normalized[m.start() :].strip()


def _normalize_whitespace(text: str) -> str:
    text = _MID_LINE_BREAK_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def _strip_with_offset(text: str, base_offset: int) -> tuple[str, int, int]:
    """.strip() dịch chuyển start/end so với text gốc (bỏ whitespace 2 đầu) —
    hàm này cắt VÀ trả kèm (char_start, char_end) TUYỆT ĐỐI (tính theo
    `base_offset` = vị trí bắt đầu của `text` CHƯA cắt trong main_text), để
    B0 sau này biết chính xác quote có vắt qua ranh giới Khoan/Dieu không
    (đo offset, không suy luận qua similarity score mờ như bản cũ)."""
    stripped = text.strip()
    if not stripped:
        return stripped, base_offset, base_offset
    lstrip_len = len(text) - len(text.lstrip())
    abs_start = base_offset + lstrip_len
    return stripped, abs_start, abs_start + len(stripped)


def _split_khoan(dieu_id: str, dieu_text: str, dieu_text_base_offset: int) -> list[Khoan]:
    """Tách phần thân 1 Điều (sau dòng tiêu đề) thành các Khoản "1. ", "2. "...

    `dieu_text_base_offset`: vị trí tuyệt đối của `dieu_text` (đã .strip())
    trong main_text — để char_start/char_end của Khoan tính theo main_text,
    không phải theo dieu_text cục bộ.

    Không có match -> trả [] (Điều không chia Khoản, vd Điều 1 câu đơn)."""
    matches = list(_KHOAN_RE.finditer(dieu_text))
    if not matches:
        return []

    khoan: list[Khoan] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(dieu_text)
        raw = dieu_text[start:end]
        body, abs_start, abs_end = _strip_with_offset(raw, dieu_text_base_offset + start)
        if not body:
            continue
        so = m.group("so")
        khoan.append(
            Khoan(khoan_id=f"{dieu_id}_{so}", khoan_so=so, text=body, char_start=abs_start, char_end=abs_end)
        )
    return khoan


def parse_document(context_id: str, passage: str) -> ParsedDoc:
    """Tách 1 văn bản (context_id + passage) thành cây Điều/Khoản.

    Trước khi tách Điều: cắt đuôi hành chính (Nơi nhận/Phụ lục) ra
    `phu_luc_raw` (xem _split_tail) — không cắt sẽ dính nội dung phụ lục
    (mẫu đơn, bảng biểu) vào Điều cuối cùng và bị _KHOAN_RE bắt nhầm số
    thứ tự bảng thành Khoản giả.

    Rỗng/toàn khoảng trắng -> dieu=[] (20 văn bản corpus rỗng hoàn toàn —
    xem DATA_NOTES.md mục 3).
    Fallback (có nội dung nhưng không match "Điều N."): 1 Dieu duy nhất
    chứa toàn văn, dieu_so="", parse_status="fallback" — không crash."""
    if not passage or not passage.strip():
        return ParsedDoc(parse_status="fallback", dieu=[], phu_luc_raw=None)

    normalized = _normalize_whitespace(passage)
    main_text, phu_luc_raw = _split_tail(normalized)
    matches = list(_DIEU_RE.finditer(main_text))

    if not matches:
        dieu = [
            Dieu(
                dieu_id=f"{context_id}_",
                dieu_so="",
                dieu_tieu_de="",
                text=main_text,
                char_start=0,
                char_end=len(main_text),
                khoan=[],
            )
        ]
        return ParsedDoc(parse_status="fallback", dieu=dieu, phu_luc_raw=phu_luc_raw)

    result: list[Dieu] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(main_text)
        raw = main_text[start:end]
        body, abs_start, abs_end = _strip_with_offset(raw, start)
        so = m.group("so")
        tieu_de = m.group("tieu_de").strip()
        dieu_id = f"{context_id}_{so}"

        # BUG THẬT đã bắt được 17/08/2026 (7,163 unit = 1.66% corpus, 2,514 văn bản
        # = 29.5% bị ảnh hưởng): với Điều NGẮN mà toàn bộ nội dung nằm CÙNG DÒNG với
        # header (vd "Điều 2. Quyết định này có hiệu lực từ ngày ký ban hành."),
        # `_DIEU_RE` hút hết nội dung vào group "tieu_de" -> `body` RỖNG -> unit này
        # đi xuống B2/B4/B6 với text rỗng (và embedding Layer 2 bị tính trên chuỗi
        # rỗng = vector rác). Đây đúng là loại Điều hay bị hỏi ("hiệu lực từ ngày
        # nào", "ai chịu trách nhiệm thi hành"). Sửa: body rỗng -> dùng tiêu đề làm
        # text, vì lúc đó tiêu đề CHÍNH LÀ nội dung. Không đổi text của Điều bình
        # thường (tránh phải tính lại toàn bộ 432k embedding).
        if not body and tieu_de:
            body = tieu_de
            abs_start, abs_end = m.start("tieu_de"), m.end("tieu_de")

        result.append(
            Dieu(
                dieu_id=dieu_id,
                dieu_so=so,
                dieu_tieu_de=tieu_de,
                text=body,
                char_start=abs_start,
                char_end=abs_end,
                khoan=_split_khoan(dieu_id, body, abs_start),
            )
        )
    return ParsedDoc(parse_status="matched", dieu=result, phu_luc_raw=phu_luc_raw)
