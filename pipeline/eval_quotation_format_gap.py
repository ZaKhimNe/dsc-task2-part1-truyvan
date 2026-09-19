"""
P2 BƯỚC 1 + 1b — Đo khoảng cách định dạng của Khối 2.

CÂU HỎI: gold trích luật kèm dòng tiêu đề Điều và số thứ tự Khoản. B đang gửi
Khoản trần. Định dạng nào khớp gold nhất?

CÁCH ĐO: với mỗi câu nối được citation (data123 -> khoan_id thật trong corpus),
so METEOR giữa
  - reference = Khối 2 của reference_answer (đoạn trích luật trong đáp án chuẩn)
  - prediction = text Khoản đó, định dạng theo từng variant
Không chạy retrieval, không chạy model. Chỉ CPU, ~20 phút.

TÁM VARIANT ĐO (định nghĩa ở src/b6_context_package/format_unit.py):
  Khoản đơn : F0 (hiện tại) · F1 · F2 · F3 · F4 · F5
  Chặn tiêu đề dài : F2 với max_title_syllables ∈ {20, 30}
  Dải Khoản  : G0 (hiện tại) · G1 · G2   — chỉ đo trên câu gold trích ≥2 Khoản

NGƯỠNG ĐẶT TRƯỚC (plan v2 §7): variant thắng phải có **mean** > 0,7721 (mức F2
đã đo). Dùng mean, không dùng median — mean nhạy với câu khó hơn.

CHÚ Ý VỀ SỐ SO SÁNH ĐƯỢC
Script này tự tách Khối 2 ra khỏi reference_answer bằng marker (xem
`split_reference_blocks`). Cách tách ở đây KHÁC `blocks.py` của part2 ở hai chỗ,
đều là sửa lỗi đã đo (plan v2 §6 mục 6):
  1. BỎ "Đồng thời" khỏi danh sách marker kết luận — chữ đó thường mở đầu KHỐI
     TRÍCH DẪN THỨ HAI, không phải kết luận.
  2. Tìm marker ở lần xuất hiện CUỐI CÙNG, không phải lần đầu.
Vì vậy con số F0 tuyệt đối ở đây có thể lệch nhẹ so với 0,7225 trong plan §1.5.
**Quyết định dựa trên CHÊNH LỆCH giữa các variant** (cùng một reference cho mọi
variant) nên lệch mốc không ảnh hưởng kết luận. Bảng in ra luôn kèm F0 để đối chiếu.

Chạy:  PYTHONIOENCODING=utf-8 python pipeline/eval_quotation_format_gap.py
"""
from __future__ import annotations

import json
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b6_context_package.format_unit import format_khoan_range, format_single_khoan

# --- Tách Khối 2 khỏi reference_answer -------------------------------------

# "Đồng thời" CỐ Ý không có trong danh sách — xem docstring module.
CONCLUSION_MARKERS = (
    "Như vậy",
    "Theo đó",
    "Do đó",
    "Vì vậy",
    "Từ đó",
    "Như đã phân tích",
    "Tóm lại",
)
_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(" + "|".join(re.escape(m) for m in CONCLUSION_MARKERS) + r")\b"
)


def split_reference_blocks(answer: str) -> tuple[str, str, str]:
    """(lead, quotation, conclusion). Trả ("", "", "") nếu không tách được.

    lead = dòng đầu không rỗng. conclusion = từ marker CUỐI CÙNG tới hết.
    quotation = phần giữa.
    """
    text = (answer or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return "", "", ""

    nl = text.find("\n")
    if nl == -1:
        return "", "", ""
    lead = text[:nl].strip()
    rest_start = nl
    while rest_start < len(text) and text[rest_start] in "\n \t":
        rest_start += 1

    matches = list(_MARKER_RE.finditer(text, rest_start))
    if not matches:
        return "", "", ""
    cut = matches[-1].start(1)
    quotation = text[rest_start:cut].strip()
    conclusion = text[cut:].strip()
    if not (lead and quotation and conclusion):
        return "", "", ""
    return lead, quotation, conclusion


# --- Nối citation của C vào corpus thật của B ------------------------------

ARTICLE_NUM_RE = re.compile(r"(\d+[a-zA-Z]?)")
CLAUSE_NUM_RE = re.compile(r"(\d+)")


def load_data123_records() -> dict[str, dict]:
    """{qid: item} từ cả 3 split. item có oracle_context.citation_metadata."""
    if not config.DATA123_DIR.exists():
        raise SystemExit(
            f"KHONG TIM THAY data123 tai: {config.DATA123_DIR}\n"
            "Script nay can citation_metadata do C trich tu reference_answer.\n"
            "Kiem lai config.REPO_ROOT (hien tai = %s) hoac copy data123/ vao do."
            % config.REPO_ROOT
        )
    records: dict[str, dict] = {}
    for split in ("train", "validation", "test"):
        path = config.DATA123_DIR / split / "baseline_eligible.json"
        if not path.exists():
            print(f"  [canh bao] thieu split: {path}", flush=True)
            continue
        for item in json.loads(path.read_text(encoding="utf-8")):
            records[str(item["id"])] = item
    return records


def extract_first_num(text: str, pattern: re.Pattern) -> str | None:
    m = pattern.search(text or "")
    return m.group(1) if m else None


def link_citation(
    parsed_corpus: dict,
    doc_number_to_context: dict,
    citation_metadata: dict,
) -> tuple[str | None, dict | None, list[dict]]:
    """Nối 1 citation -> (khoan_id, dieu_record, list Khoản đã trích).

    Trả (None, None, []) nếu không nối được. `list Khoản đã trích` gồm mọi Khoản
    mà citation nhắc tới trong CÙNG Điều đầu tiên (dùng cho nhánh G — plan P8 ghi
    12,3% gold trích >1 Điều/Khoản).
    """
    doc_numbers = citation_metadata.get("document_numbers") or []
    articles = citation_metadata.get("articles") or []
    clauses = citation_metadata.get("clauses") or []
    if not doc_numbers or not articles:
        return None, None, []

    context_id = doc_number_to_context.get(doc_numbers[0])
    if not context_id:
        return None, None, []
    doc = parsed_corpus.get(context_id)
    if not doc:
        return None, None, []

    dieu_so = extract_first_num(articles[0], ARTICLE_NUM_RE)
    for dieu in doc.get("dieu", []):
        if (dieu.get("dieu_so") or "").strip() != dieu_so:
            continue
        khoan_list = dieu.get("khoan", [])
        if not khoan_list or not clauses:
            return dieu.get("dieu_id"), dieu, []
        wanted = {extract_first_num(c, CLAUSE_NUM_RE) for c in clauses}
        wanted.discard(None)
        hits = [k for k in khoan_list if (k.get("khoan_so") or "").strip() in wanted]
        if not hits:
            return None, None, []
        return hits[0]["khoan_id"], dieu, hits
    return None, None, []


# --- Đo ---------------------------------------------------------------------

SINGLE_VARIANTS = [
    ("F0", "F0", None),
    ("F1", "F1", None),
    ("F2", "F2", None),
    ("F3", "F3", None),
    ("F4", "F4", None),
    ("F5", "F5", None),
    ("F2-cap20", "F2", 20),
    ("F2-cap30", "F2", 30),
]
RANGE_VARIANTS = ["G0", "G1", "G2"]
BASELINE_MEAN = 0.7721  # nguong dat truoc: F2 da do (plan v2 §7)
N_BOOTSTRAP = 2000
SEED = 42


def _meteor():
    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError:
        raise SystemExit(
            "Thieu nltk. Cai: pip install -r requirements.txt\n"
            'Roi tai resource: python -c "import nltk; nltk.download(\'wordnet\')"'
        )
    return meteor_score


def score(meteor_score, reference: str, prediction: str) -> float:
    if not prediction.strip():
        return 0.0
    return float(meteor_score([reference.split()], prediction.split()))


def paired_bootstrap(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """CI 95% cho mean(b) - mean(a), bootstrap theo cap. (delta, lo, hi)."""
    assert len(a) == len(b)
    rng = random.Random(SEED)
    n = len(a)
    diffs = [y - x for x, y in zip(a, b)]
    delta = statistics.fmean(diffs)
    means = []
    for _ in range(N_BOOTSTRAP):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    return delta, means[int(0.025 * N_BOOTSTRAP)], means[int(0.975 * N_BOOTSTRAP)]


def main() -> None:
    meteor_score = _meteor()

    print("Dang load parsed_corpus + doc_number_index + train.json + data123...", flush=True)
    parsed_corpus = io_utils.load_parsed_corpus()
    doc_number_index = json.loads(
        (config.OUTPUTS_DIR / "doc_number_index.json").read_text(encoding="utf-8")
    )["usable"]
    train = io_utils.load_train()
    records = load_data123_records()
    print(f"  data123: {len(records)} cau", flush=True)

    # Thu thap mau do duoc
    rows = []           # (qid, ref_quote, dieu, khoan_record, hits)
    n_no_answer = 0
    n_split_fail = 0
    n_link_fail = 0
    n_dieu_only = 0

    for qid, item in records.items():
        qa = train.get(qid)
        answer = (qa or {}).get("answer")
        if not answer:
            n_no_answer += 1
            continue
        _, ref_quote, _ = split_reference_blocks(answer)
        if not ref_quote:
            n_split_fail += 1
            continue
        cm = (item.get("oracle_context") or {}).get("citation_metadata") or {}
        unit_id, dieu, hits = link_citation(parsed_corpus, doc_number_index, cm)
        if unit_id is None:
            n_link_fail += 1
            continue
        if not hits:
            n_dieu_only += 1
            continue
        rows.append((qid, ref_quote, dieu, hits[0], hits))

    print(f"\nKhong co answer trong train.json : {n_no_answer}")
    print(f"Khong tach duoc 3 khoi           : {n_split_fail}")
    print(f"Khong noi duoc citation          : {n_link_fail}")
    print(f"Noi duoc nhung chi tới cap Dieu  : {n_dieu_only}")
    print(f"DUNG DUOC (n)                    : {len(rows)}")
    if not rows:
        raise SystemExit("Khong co mau nao dung duoc — kiem lai data123 va train.json")

    # ---- Khoan don: 8 variant
    print("\n=== KHOAN DON — 8 kieu dinh dang ===")
    print(f"{'variant':<10} {'mean':>8} {'median':>8}   {'vs F0 (CI 95%)':>28}")
    per_variant: dict[str, list[float]] = {}
    for name, variant, cap in SINGLE_VARIANTS:
        vals = []
        for _qid, ref_quote, dieu, k, _hits in rows:
            pred = format_single_khoan(
                variant=variant,
                dieu_so=dieu.get("dieu_so", ""),
                dieu_tieu_de=dieu.get("dieu_tieu_de", ""),
                dieu_text=dieu.get("text", ""),
                khoan_so=k.get("khoan_so", ""),
                khoan_text=k.get("text", ""),
                max_title_syllables=cap,
            )
            vals.append(score(meteor_score, ref_quote, pred))
        per_variant[name] = vals
        mean = statistics.fmean(vals)
        med = statistics.median(vals)
        if name == "F0":
            print(f"{name:<10} {mean:>8.4f} {med:>8.4f}   {'(moc)':>28}")
        else:
            d, lo, hi = paired_bootstrap(per_variant["F0"], vals)
            flag = " *" if lo > 0 else ""
            print(f"{name:<10} {mean:>8.4f} {med:>8.4f}   {d:+.4f} [{lo:+.4f}, {hi:+.4f}]{flag}")

    best = max(per_variant.items(), key=lambda kv: statistics.fmean(kv[1]))
    best_mean = statistics.fmean(best[1])
    print(f"\nVariant tot nhat theo mean: {best[0]} = {best_mean:.4f}")
    print(f"Nguong dat truoc (F2 da do): {BASELINE_MEAN:.4f} -> "
          f"{'DAT' if best_mean > BASELINE_MEAN else 'KHONG DAT'}")

    # ---- Dai Khoan: chi tren cau gold trich >=2 Khoan cung Dieu
    multi = [r for r in rows if len(r[4]) >= 2]
    print(f"\n=== DAI KHOAN — chi tren cau gold trich >=2 Khoan (n={len(multi)}) ===")
    if len(multi) < 30:
        print("  n qua nho (<30), khong ket luan. Chay P8 de lam giau gold roi do lai.")
    else:
        print(f"{'variant':<10} {'mean':>8} {'median':>8}   {'vs G0 (CI 95%)':>28}")
        g_vals: dict[str, list[float]] = {}
        for variant in RANGE_VARIANTS:
            vals = []
            for _qid, ref_quote, dieu, _k, hits in multi:
                n_khoan_dieu = len(dieu.get("khoan", []))
                pred = format_khoan_range(
                    variant=variant,
                    dieu_so=dieu.get("dieu_so", ""),
                    dieu_tieu_de=dieu.get("dieu_tieu_de", ""),
                    dieu_text=dieu.get("text", ""),
                    khoan_items=[{"khoan_so": h.get("khoan_so", ""), "text": h.get("text", "")}
                                 for h in hits],
                    covers_whole_dieu=(len(hits) == n_khoan_dieu),
                )
                vals.append(score(meteor_score, ref_quote, pred))
            g_vals[variant] = vals
            mean = statistics.fmean(vals)
            med = statistics.median(vals)
            if variant == "G0":
                print(f"{variant:<10} {mean:>8.4f} {med:>8.4f}   {'(moc)':>28}")
            else:
                d, lo, hi = paired_bootstrap(g_vals["G0"], vals)
                flag = " *" if lo > 0 else ""
                print(f"{variant:<10} {mean:>8.4f} {med:>8.4f}   {d:+.4f} [{lo:+.4f}, {hi:+.4f}]{flag}")
        best_g = max(g_vals.items(), key=lambda kv: statistics.fmean(kv[1]))
        print(f"\nVariant dai tot nhat: {best_g[0]} = {statistics.fmean(best_g[1]):.4f}")

    print("\n(* = can duoi CI > 0, tuc thang co y nghia)")
    print("\nGHI KET QUA VAO plan §7 TRUOC KHI build V8.")
    print("Dat variant thang vao SINGLE_VARIANT / RANGE_VARIANT cua")
    print("  pipeline/build_qa_packages_public_v8_tieu_de_khoan.py")


if __name__ == "__main__":
    main()
