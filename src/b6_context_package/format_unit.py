"""
B6 — Định dạng text Khối 2 (đoạn trích luật) trước khi đóng gói cho Generator.

VÌ SAO CÓ FILE NÀY (P2 của PLAN_TOI_THUONG_RETRIEVAL_v2.md §4)

Gold trích luật KÈM dòng tiêu đề Điều và số thứ tự Khoản; B đang bỏ cả hai, nên
Khối 2 mất điểm METEOR ngay cả khi truy hồi ĐÚNG Khoản. Đo được:

  - offline trên n=1.427 câu nối được citation (`eval_quotation_format_gap.py`):
    F0 (Khoản trần) mean 0,7225 -> F2 (tiêu đề + số Khoản) mean 0,7721 = +0,0496
  - đo METEOR TOÀN BÀI trên 188 câu heldout retrieval lấy đúng Khoản gold:
    mốc 3 đoạn thô 0,7559 -> thêm tiêu đề + số Khoản 0,8041 (+0,0482)
    -> + gộp Khoản liền kề 0,8798 (+0,1239)

Module này là NGUỒN DUY NHẤT của logic định dạng, dùng chung cho:
  - `pipeline/eval_quotation_format_gap.py` (đo offline, chọn variant thắng)
  - `pipeline/build_qa_packages_public_v8_tieu_de_khoan.py` (build bản nộp)
  - `pipeline/build_finetune_pairs.py` (P6 — positive phải khớp inference)

KHÔNG copy-paste logic này sang chỗ khác. Train lệch inference là 1 trong 3 lỗi
gốc khiến Team IR fine-tune âm 2/2 lần (xem plan §4 P6).

SCHEMA parsed_corpus.jsonl (output B1, xác nhận trên data thật 18/09):
  record : {context_id, name, link, parse_status, dieu[], phu_luc_raw}
  dieu   : {dieu_id, dieu_so, dieu_tieu_de, text, char_start, char_end, khoan[]}
  khoan  : {khoan_id, khoan_so, text, char_start, char_end}

Hai điểm quan trọng của schema:
  - `khoan["text"]` KHÔNG chứa tiền tố "1." — phải tự thêm khi muốn (F2 trở lên).
  - `dieu["text"]` CÓ chứa "1. ... 2. ..." (toàn bộ Điều đã ghép sẵn).

BA CON SỐ ĐO TRÊN CORPUS THẬT (3.000 văn bản đầu, 44.962 Điều, 18/09):
  - 1,1% Điều KHÔNG có `dieu_tieu_de`. (Plan §1.6(a) ghi "100% Điều có" — sai
    nhẹ; mọi hàm ở đây thoái về F0 khi thiếu tiêu đề.)
  - 5,6% Điều có `text == dieu_tieu_de` (nội dung bị hút hết vào tiêu đề).
    Suy ra toàn corpus ≈ 7.170 Điều, khớp con số 7.163 mà
    `patch_layer2_embeddings.py` từng vá. Chặn ở `_should_prefix_title()`.
  - Độ dài tiêu đề: median 9 âm tiết, mean 11,6, p90 22, p99 52, max 245.
    11,5% dài hơn 20 âm tiết. Đuôi dài là các Điều của văn bản SỬA ĐỔI, tiêu đề
    kiểu "Sửa đổi, bổ sung một số điều của Thông tư số 12/2020/TT-BGTVT ngày 29
    tháng 5 năm 2020 của Bộ trưởng..." — đó là mô tả văn bản, không phải chủ đề
    của Điều, nên nhiều khả năng chỉ thêm nhiễu.
    -> `max_title_syllables` cho phép chặn. MẶC ĐỊNH None (không chặn) để
    `eval_quotation_format_gap.py` đo xem chặn có lợi hay không. KHÔNG tự đặt
    ngưỡng theo cảm giác — để số liệu quyết định (plan §0 quy tắc 4).
"""
from __future__ import annotations

from typing import Literal

# Các kiểu định dạng cho unit là 1 KHOẢN đơn.
# F0 = hiện tại (Khoản trần). F1/F2 đã đo. F3/F4/F5 là P2 bước 1 cần đo.
SingleVariant = Literal["F0", "F1", "F2", "F3", "F4", "F5"]
# Các kiểu định dạng cho DẢI nhiều Khoản liền kề (P2 bước 1b).
RangeVariant = Literal["G0", "G1", "G2"]

ELLIPSIS_LINE = "…"


def _norm(text: str) -> str:
    return (text or "").strip()


def _syl(text: str) -> int:
    return len(_norm(text).split())


def _usable_title(
    dieu_text: str,
    dieu_tieu_de: str,
    max_title_syllables: int | None,
) -> str:
    """Tiêu đề dùng được, hoặc "" khi phải bỏ.

    Ba lý do bỏ, theo thứ tự kiểm:
      1. Không có tiêu đề (1,1% Điều).
      2. `text == dieu_tieu_de` (5,6% Điều) — prefix thêm sẽ tạo dòng lặp.
      3. Tiêu đề dài hơn `max_title_syllables` (chỉ khi tham số được đặt).
    """
    td = _norm(dieu_tieu_de)
    if not td:
        return ""
    if _norm(dieu_text) == td:
        return ""
    if max_title_syllables is not None and _syl(td) > max_title_syllables:
        return ""
    return td


def _dieu_header(dieu_so: str, title: str, with_dieu_prefix: bool) -> str:
    """Dòng tiêu đề. `with_dieu_prefix=True` -> "Điều 12. {tiêu đề}" (gold kiểu B,
    ~13%); False -> "{tiêu đề}" trần (gold kiểu A, ~87%)."""
    if not title:
        return ""
    if with_dieu_prefix and _norm(dieu_so):
        return f"Điều {_norm(dieu_so)}. {title}"
    return title


def _numbered(khoan_so: str, text: str) -> str:
    """"1. {text}" — khoan["text"] không có tiền tố số nên phải tự thêm."""
    ks = _norm(khoan_so)
    return f"{ks}. {_norm(text)}" if ks else _norm(text)


def format_single_khoan(
    *,
    variant: SingleVariant,
    dieu_so: str,
    dieu_tieu_de: str,
    dieu_text: str,
    khoan_so: str,
    khoan_text: str,
    max_title_syllables: int | None = None,
) -> str:
    """Định dạng 1 Khoản đơn theo variant.

    | Mã | Kết quả                                          |
    |----|--------------------------------------------------|
    | F0 | {text}                                           |
    | F1 | {tiêu đề}\\n{text}                                |
    | F2 | {tiêu đề}\\n{số}. {text}                          |
    | F3 | Điều {N}. {tiêu đề}\\n{số}. {text}                |
    | F4 | {tiêu đề}\\n…\\n{số}. {text}                       |
    | F5 | như F4 (tầng gọi quyết định dải có phủ hết Điều)  |

    Khi tiêu đề không dùng được (xem `_usable_title`), F1 thoái về F0 còn
    F2–F5 vẫn giữ số Khoản — số Khoản độc lập với tiêu đề và luôn có trong gold.
    """
    body = _norm(khoan_text)
    if variant == "F0":
        return body

    title = _usable_title(dieu_text, dieu_tieu_de, max_title_syllables)
    if not title:
        return body if variant == "F1" else _numbered(khoan_so, body)

    header = _dieu_header(dieu_so, title, with_dieu_prefix=(variant == "F3"))

    if variant == "F1":
        return f"{header}\n{body}"
    if variant in ("F2", "F3"):
        return f"{header}\n{_numbered(khoan_so, body)}"
    if variant in ("F4", "F5"):
        return f"{header}\n{ELLIPSIS_LINE}\n{_numbered(khoan_so, body)}"
    raise ValueError(f"variant không hợp lệ: {variant}")


def format_khoan_range(
    *,
    variant: RangeVariant,
    dieu_so: str,
    dieu_tieu_de: str,
    dieu_text: str,
    khoan_items: list[dict],
    covers_whole_dieu: bool,
    max_title_syllables: int | None = None,
) -> str:
    """Định dạng DẢI Khoản liền kề cùng Điều (kết quả `gop_khoan_lien_ke`).

    `khoan_items` = list `{khoan_so, text}` theo đúng thứ tự trong Điều.
    `covers_whole_dieu` = dải có phủ TOÀN BỘ Khoản của Điều hay không.

    | Mã | Kết quả                                                       |
    |----|---------------------------------------------------------------|
    | G0 | các text nối bằng "\\n\\n" (hiện tại, không tiêu đề, không số)  |
    | G1 | {tiêu đề}\\n{lo}. {text}\\n{lo+1}. {text}…                      |
    | G2 | {tiêu đề}\\n…\\n{lo}. {text}…  (36% gold kiểu A có dòng "…")    |

    G1/G2 khi `covers_whole_dieu=True` dùng header "Điều {N}. {tiêu đề}" (gold
    kiểu B trích cả Điều thì có tiền tố "Điều N."), ngược lại dùng tiêu đề trần.
    Dòng "…" chỉ thêm khi dải KHÔNG phủ hết Điều (có phần bị lược thật).
    """
    numbered = [
        _numbered(k.get("khoan_so", ""), _norm(k.get("text", "")))
        for k in khoan_items
        if _norm(k.get("text", ""))
    ]
    if not numbered:
        return ""

    if variant == "G0":
        return "\n\n".join(
            _norm(k.get("text", "")) for k in khoan_items if _norm(k.get("text", ""))
        )

    title = _usable_title(dieu_text, dieu_tieu_de, max_title_syllables)
    if not title:
        return "\n".join(numbered)

    header = _dieu_header(dieu_so, title, with_dieu_prefix=covers_whole_dieu)
    lines = [header]
    if variant == "G2" and not covers_whole_dieu:
        lines.append(ELLIPSIS_LINE)
    lines.extend(numbered)
    return "\n".join(lines)


def find_dieu_for_unit(parsed_corpus: dict, context_id: str, unit_id: str):
    """Trả `dieu` record chứa `unit_id` (khoan_id hoặc dieu_id), hoặc None.

    Dùng chung cho cả script đo và script build, để hai bên không lệch cách tra.
    """
    doc = parsed_corpus.get(context_id)
    if not doc:
        return None
    for dieu in doc.get("dieu", []):
        if dieu.get("dieu_id") == unit_id:
            return dieu
        for k in dieu.get("khoan", []):
            if k.get("khoan_id") == unit_id:
                return dieu
    return None


def title_for_unit(
    parsed_corpus: dict,
    context_id: str,
    unit_id: str,
    max_title_syllables: int | None = None,
) -> str:
    """Tiêu đề Điều của unit — dùng để điền `article_title` trong ContextItem
    (Generator có thể dùng ở Lead, xem plan §6 mục 4). "" nếu không tra được
    hoặc tiêu đề không dùng được."""
    dieu = find_dieu_for_unit(parsed_corpus, context_id, unit_id)
    if dieu is None:
        return ""
    return _usable_title(
        dieu.get("text", ""), dieu.get("dieu_tieu_de", ""), max_title_syllables
    )
