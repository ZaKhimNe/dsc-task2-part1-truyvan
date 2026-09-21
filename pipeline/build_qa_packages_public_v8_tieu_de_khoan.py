"""
Build QA packages V8 — tiêu đề Điều + số Khoản + gộp Khoản liền kề.
(P2 của PLAN_TOI_THUONG_RETRIEVAL_v2.md §4 — LƯỢT C ĐẦU TIÊN)

COPY TỪ `build_qa_packages_public_v6_adaptive_dedup.py`, đổi ĐÚNG 5 chỗ:

  1. Text Khối 2 đi qua `src/b6_context_package/format_unit.py` — thêm dòng tiêu
     đề Điều và số thứ tự Khoản. Đây là thay đổi chính.
  2. `MAX_CONTEXTS = 3` (V6 dùng 6). Lý do ở plan v2 §1.7: sau khi Khối 2 đủ chữ,
     thêm đoạn rác làm GIẢM điểm.
  3. `PER_ITEM_BUDGET_SYLLABLES` 250 -> 350, để tiêu đề (median 9 âm tiết) không
     ăn vào chỗ của phần gộp Khoản liền kề (median 41 âm tiết, plan v2 §1.8).
  4. Ngân sách gộp TRỪ TRƯỚC độ dài tiêu đề, và mọi phép đếm âm tiết tính trên
     text ĐÃ định dạng — nếu không sẽ vượt ngân sách mà không biết.
  5. Jaccard dedup tính trên text KHÔNG gồm tiêu đề. Hai Khoản khác nhau cùng
     một Điều dùng chung tiêu đề -> Jaccard tăng giả -> dedup xoá oan.

Thêm trường `article_title` vào mỗi context (mặc định "" nếu Điều không có tiêu
đề dùng được).

--------------------------------------------------------------------------------
BỔ SUNG 19/09 — THAM SỐ HOÁ ĐỂ BUILD V9 CÙNG MỘT ĐƯỜNG CODE

KHÔNG tạo file thứ hai cho V9. V8 và V9 phải khác ĐÚNG MỘT BIẾN, mà nhân đôi
code là cách chắc chắn nhất để hai bản trôi ra xa nhau — bẫy H4 của Task 1 ("hai
artefact mô tả hai cái rổ mà tên file không báo"). Nên mọi thứ V9 cần đều là
tham số dòng lệnh, và MẶC ĐỊNH BẰNG ĐÚNG HÀNH VI V8 CŨ:

    python pipeline/build_qa_packages_public_v8_tieu_de_khoan.py
        -> tái hiện V8 cũ (alpha=0 nên KHÔNG xếp lại, giữ nguyên thứ tự file)

    python pipeline/build_qa_packages_public_v8_tieu_de_khoan.py \
        --rerank-path outputs/layer3/L3_rerank_public_title.jsonl \
        --alpha 0.7 --max-contexts 2 --per-item 200 --expected-median 450
        -> cấu hình V9 chốt ngày 19/09

TRỘN THỨ HẠNG (RRF có trọng số), dùng khi --alpha > 0:

    s = alpha/(RRF_K + dense_rank) + (1-alpha)/(RRF_K + rerank_rank)

  `rerank_rank` là vị trí trong `ranked_units` (file đã sắp theo điểm reranker).
  Cả hai hạng đếm từ 0 — `dense_rank` trong file cũng đếm từ 0.

  Dạng NGHỊCH ĐẢO chứ không trung bình tuyến tính, vì chỉ đỉnh danh sách có giá
  trị (sản xuất giao 2 đoạn). Trung bình tuyến tính coi khoảng cách hạng 1↔2
  bằng khoảng cách hạng 49↔50, nên ứng viên hạng 1 ở bảng này và hạng 50 ở bảng
  kia sẽ ra hạng 25 và biến mất khỏi top-2.

  alpha=1 dense thuần, alpha=0 rerank thuần. Đo trên mẫu train seed 1909: đỉnh
  phẳng trong dải 0,6–0,8; kiểm chéo hai nửa chọn 0,6 và 0,8 nên lấy trung điểm
  0,7. Lợi ích ước lượng bằng KIỂM CHÉO là +0,006 — KHÔNG phải +0,0082 của
  bootstrap toàn mẫu; chênh lệch đó chính là độ thiên lệch do chọn bằng mắt.

VỀ `row` VÀ BUG MÃ TRÙNG: script này tra text bằng `get_unit_text(..., unit_id)`,
KHÔNG dùng `row`. Nên cảnh báo "60–64% ứng viên có row = -1" KHÔNG áp vào đây —
không có đường nào để hỏng. Đổi lại nó chịu phơi nhiễm mã trùng: 7.631 unit_id
trùng, nhưng E1 đo được 100% bản sao nằm CÙNG MỘT ĐIỀU, và phần mở rộng ngân
sách vốn gộp Khoản liền kề cùng Điều, nên khoản anh em lấy nhầm nhiều khả năng
vẫn nằm trong đoạn giao đi. Script in tỉ lệ đó ra mỗi lần chạy thay vì để nó vô
hình.

ĐỘ DÀI TỔNG MỚI LÀ BIẾN THẬT, không phải --per-item. Đo 19/09 trên mẫu 1909:
k=1 tăng đều theo ngân sách, k=2 có đỉnh ở 200 (tổng 450 âm tiết), k=3 giảm đều
— cả ba cùng chỉ về tổng 450–500. `TARGET_TOTAL_SYLLABLES` vẫn 535 nhưng ở cấu
hình k=2/per_item=200 nó không còn bị chạm tới, nên --per-item thành nút cắt.
Vì thế PHẢI đọc dòng "Do dai TONG" sau build chứ đừng tin tham số.
--------------------------------------------------------------------------------

TRƯỚC KHI CHẠY: chạy `pipeline/eval_quotation_format_gap.py`, ghi variant thắng
vào `SINGLE_VARIANT` / `RANGE_VARIANT`. Mặc định F2/G1 là variant tốt nhất ĐÃ ĐO
(F2 mean 0,7721 trên n=1.427). F3/F4/F5/G2 chưa đo; dự đoán từ thống kê gold:
F3 kỳ vọng âm (gold kiểu B có tiền tố "Điều N." chỉ ~13%, kiểu A ~87%), còn
F4/F5/G2 chênh đúng một token "…" nên dưới sàn nhiễu.

Chạy: PYTHONIOENCODING=utf-8 python pipeline/build_qa_packages_public_v8_tieu_de_khoan.py
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b2_retrieval.retrieve import get_unit_text
from src.b6_context_package.format_unit import (
    find_dieu_for_unit,
    format_khoan_range,
    format_single_khoan,
)
from src.b6_context_package.package import (
    build_context_item,
    build_qa_package,
    write_qa_packages_json,
)

# --- Mặc định = HÀNH VI V8 CŨ. Đổi bằng dòng lệnh, đừng sửa ở đây. ------------
DEFAULT_RERANK_PATH = Path("outputs/layer3/L3_rerank_public.jsonl")
DEFAULT_OUT_PATH = config.OUTPUTS_DIR / "qa_packages_public_v8_tieu_de_khoan.json"

DEFAULT_ALPHA = 0.0            # 0 = giữ nguyên thứ tự file (rerank thuần) = V8 cũ
DEFAULT_MAX_CONTEXTS = 3
DEFAULT_PER_ITEM = 350
RRF_K = 60                     # hằng số RRF chuẩn; nhỏ hơn = dồn trọng số về đỉnh

# --- Variant định dạng: ĐIỀN TỪ KẾT QUẢ eval_quotation_format_gap.py ---------
SINGLE_VARIANT = "F2"
RANGE_VARIANT = "G1"
MAX_TITLE_SYLLABLES = None  # None = không chặn. Đặt 20/30 nếu eval cho thấy có lợi.

TARGET_TOTAL_SYLLABLES = 535
LEN_CAP_SYLLABLES = 1200
JACCARD_DEDUP_THRESHOLD = 0.7

V6_MEDIAN_SYLLABLES = 620
MAX_ALLOWED_MEDIAN_SHIFT = 90


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build QA packages V8/V9 — cùng một đường code, khác tham số."
    )
    p.add_argument("--rerank-path", type=Path, default=DEFAULT_RERANK_PATH,
                   help=f"File kết quả Layer 3 (mặc định {DEFAULT_RERANK_PATH})")
    p.add_argument("--out", type=Path, default=None,
                   help="File gói ra. Bỏ trống: tên V8 cũ nếu cấu hình mặc định, "
                        "ngược lại tên tự suy từ cấu hình.")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                   help="Trọng số nhánh dense khi trộn hạng. 1=dense thuần, "
                        f"0=rerank thuần (mặc định {DEFAULT_ALPHA} = V8 cũ).")
    p.add_argument("--max-contexts", type=int, default=DEFAULT_MAX_CONTEXTS,
                   help=f"Số đoạn tối đa giao đi (mặc định {DEFAULT_MAX_CONTEXTS})")
    p.add_argument("--per-item", type=int, default=DEFAULT_PER_ITEM,
                   help=f"Ngân sách âm tiết mỗi đoạn (mặc định {DEFAULT_PER_ITEM})")
    p.add_argument("--questions", choices=["public", "train", "private"], default="public",
                   help="Nguồn câu hỏi. 'train' để build gói ĐO OFFLINE — train.json "
                        "có reference_answer nên kernel tự chấm được, không cần LB.")
    p.add_argument("--expected-median", type=int, default=None,
                   help="Median độ dài TỔNG mong đợi, để kiểm sau build. Bỏ trống "
                        f"thì so với mốc V6 ({V6_MEDIAN_SYLLABLES}).")
    args = p.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        p.error("--alpha phai nam trong [0, 1]")
    if args.max_contexts < 1:
        p.error("--max-contexts phai >= 1")
    if args.per_item < 1:
        p.error("--per-item phai >= 1")
    return args


def resolve_out_path(args: argparse.Namespace) -> Path:
    """Tên file ra SUY TỪ cấu hình, không đặt tay — để không thể dán nhãn "V8"
    lên một bản build từ rổ khác. Chỉ giữ tên V8 cũ khi cấu hình đúng mặc định."""
    if args.out is not None:
        return args.out
    is_legacy = (
        args.rerank_path == DEFAULT_RERANK_PATH
        and args.alpha == DEFAULT_ALPHA
        and args.max_contexts == DEFAULT_MAX_CONTEXTS
        and args.per_item == DEFAULT_PER_ITEM
        and args.questions == "public"
    )
    if is_legacy:
        return DEFAULT_OUT_PATH
    # `--questions` vào tên dù stem rổ train/public vốn đã khác nhau — đừng dựa
    # vào may. Gói đo offline và gói nộp bài không được phép trùng tên.
    return config.OUTPUTS_DIR / (
        f"qa_packages_{args.questions}_{args.rerank_path.stem}_a{args.alpha:g}"
        f"_k{args.max_contexts}_b{args.per_item}.json"
    )


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blended_rank(ranked_units: list[dict], alpha: float) -> list[dict]:
    """Xếp lại `ranked_units` bằng RRF có trọng số.

    alpha == 0 -> TRẢ NGUYÊN danh sách, không đụng gì. Đây là đường V8 cũ, giữ
    riêng để bản mặc định tái hiện được, không phụ thuộc chuyện sắp xếp có ổn
    định hay không.
    """
    if alpha == 0.0:
        return ranked_units

    n_missing = sum(1 for u in ranked_units if u.get("dense_rank") is None)
    if n_missing:
        raise SystemExit(
            f"--alpha={alpha} can truong 'dense_rank' nhung {n_missing}/"
            f"{len(ranked_units)} ung vien khong co. File rerank cu khong mang "
            f"truong nay — dung file co dense_rank, hoac chay voi --alpha 0."
        )

    scored = []
    for rerank_rank, u in enumerate(ranked_units):
        s = alpha / (RRF_K + u["dense_rank"]) + (1.0 - alpha) / (RRF_K + rerank_rank)
        scored.append((s, rerank_rank, u))
    scored.sort(key=lambda t: (-t[0], t[1]))   # hoà thì giữ thứ tự cũ
    return [u for _, _, u in scored]


def infer_unit_type(unit_id: str, context_id: str) -> str:
    suffix = unit_id[len(context_id) + 1 :] if unit_id.startswith(context_id + "_") else ""
    n_parts = len([p for p in suffix.split("_") if p])
    if n_parts >= 2:
        return "khoan"
    if n_parts == 1:
        return "dieu_fallback"
    return "doc_fallback"


def syl_count(text: str) -> int:
    return len(text.split())


def normalize_tokens(text: str) -> set[str]:
    return set(" ".join(text.split()).lower().split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_dup_id_set(parsed_corpus: dict) -> set[str]:
    """unit_id xuất hiện nhiều hơn một lần trong corpus. E1 đo được 7.631 id như
    vậy, 100% bản sao nằm cùng một Điều. Dùng để BÁO CÁO phơi nhiễm, không sửa."""
    counts: dict[str, int] = {}
    for doc in parsed_corpus.values():
        for d in doc.get("dieu", []):
            uid = d.get("dieu_id")
            if uid:
                counts[uid] = counts.get(uid, 0) + 1
            for k in d.get("khoan", []):
                kid = k.get("khoan_id")
                if kid:
                    counts[kid] = counts.get(kid, 0) + 1
    return {uid for uid, c in counts.items() if c > 1}


def chon_dai_khoan(dieu: dict, khoan_id: str, budget: int) -> tuple[list[dict], bool, bool]:
    """Mở rộng từ Khoản đã chọn sang Khoản liền kề CÙNG ĐIỀU (tới/lui) cho tới
    khi chạm `budget` âm tiết. `budget` là ngân sách cho PHẦN THÂN — phần tiêu đề
    đã bị trừ ở tầng gọi.

    Trả (danh_sách_Khoản_theo_thứ_tự, phủ_hết_Điều, có_mở_rộng).
    """
    khoan_list = dieu.get("khoan", [])
    idx = next(
        (i for i, k in enumerate(khoan_list) if k.get("khoan_id") == khoan_id), None
    )
    if idx is None:
        return [], False, False

    lo = hi = idx
    total = syl_count(khoan_list[idx].get("text", ""))
    while total < budget:
        can_hi = hi + 1 < len(khoan_list)
        can_lo = lo - 1 >= 0
        if not can_hi and not can_lo:
            break
        if can_hi:
            hi += 1
            total += syl_count(khoan_list[hi].get("text", ""))
        elif can_lo:
            lo -= 1
            total += syl_count(khoan_list[lo].get("text", ""))

    items = khoan_list[lo : hi + 1]
    return items, len(items) == len(khoan_list), (lo, hi) != (idx, idx)


def decide_text(
    u: dict, parsed_corpus: dict, per_item_budget: int
) -> tuple[str, str, str, bool]:
    """Trả (text_đã_định_dạng, text_thân_không_tiêu_đề, article_title, có_gộp).

    `text_thân_không_tiêu_đề` dùng cho Jaccard dedup — xem điểm 5 docstring module.
    """
    dieu = find_dieu_for_unit(parsed_corpus, u["context_id"], u["unit_id"])
    if dieu is None:
        return u["text"], u["text"], "", False

    dieu_so = dieu.get("dieu_so", "")
    dieu_tieu_de = dieu.get("dieu_tieu_de", "")
    dieu_text = dieu.get("text", "")

    if u["unit_type"] != "khoan":
        formatted = format_single_khoan(
            variant=SINGLE_VARIANT if SINGLE_VARIANT != "F0" else "F0",
            dieu_so=dieu_so,
            dieu_tieu_de=dieu_tieu_de,
            dieu_text=dieu_text,
            khoan_so="",
            khoan_text=u["text"],
            max_title_syllables=MAX_TITLE_SYLLABLES,
        )
        title = formatted[: len(formatted) - len(u["text"])].strip()
        return formatted, u["text"], title, False

    # Trừ trước độ dài tiêu đề khỏi ngân sách thân (điểm 4 docstring module)
    title_syl = 0
    if SINGLE_VARIANT != "F0" and dieu_tieu_de and dieu_text.strip() != dieu_tieu_de.strip():
        if MAX_TITLE_SYLLABLES is None or syl_count(dieu_tieu_de) <= MAX_TITLE_SYLLABLES:
            title_syl = syl_count(dieu_tieu_de)
    body_budget = max(per_item_budget - title_syl, 1)

    items, covers_whole, merged = chon_dai_khoan(dieu, u["unit_id"], body_budget)
    if not items:
        return u["text"], u["text"], "", False

    khoan_items = [
        {"khoan_so": k.get("khoan_so", ""), "text": k.get("text", "")} for k in items
    ]
    body_only = "\n\n".join(k["text"] for k in khoan_items if k["text"].strip())

    if len(items) == 1:
        formatted = format_single_khoan(
            variant=SINGLE_VARIANT,
            dieu_so=dieu_so,
            dieu_tieu_de=dieu_tieu_de,
            dieu_text=dieu_text,
            khoan_so=khoan_items[0]["khoan_so"],
            khoan_text=khoan_items[0]["text"],
            max_title_syllables=MAX_TITLE_SYLLABLES,
        )
    else:
        formatted = format_khoan_range(
            variant=RANGE_VARIANT,
            dieu_so=dieu_so,
            dieu_tieu_de=dieu_tieu_de,
            dieu_text=dieu_text,
            khoan_items=khoan_items,
            covers_whole_dieu=covers_whole,
            max_title_syllables=MAX_TITLE_SYLLABLES,
        )

    title = ""
    if dieu_tieu_de and dieu_text.strip() != dieu_tieu_de.strip():
        if MAX_TITLE_SYLLABLES is None or syl_count(dieu_tieu_de) <= MAX_TITLE_SYLLABLES:
            title = dieu_tieu_de.strip()
    return formatted, body_only, title, merged


def main() -> None:
    args = parse_args()
    rerank_path = args.rerank_path
    out_path = resolve_out_path(args)

    if not rerank_path.exists():
        raise SystemExit(f"Khong thay file rerank: {rerank_path}")
    rerank_md5 = file_md5(rerank_path)

    print("=== CAU HINH ===", flush=True)
    print(f"  rerank : {rerank_path}", flush=True)
    print(f"  md5    : {rerank_md5}", flush=True)
    print(f"  cau hoi: {args.questions}", flush=True)
    print(f"  alpha  : {args.alpha}  (1=dense thuan, 0=rerank thuan, RRF_K={RRF_K})", flush=True)
    print(f"  k      : {args.max_contexts}   per_item: {args.per_item}   "
          f"target_total: {TARGET_TOTAL_SYLLABLES}", flush=True)
    print(f"  ra     : {out_path}", flush=True)

    print("\nDang load parsed_corpus + public-official.json + ket qua Layer 3...", flush=True)
    parsed_corpus = io_utils.load_parsed_corpus()
    qa_src = (
        io_utils.load_train() if args.questions == "train"
        else io_utils.load_private_official() if args.questions == "private"
        else io_utils.load_public_official()
    )

    with open(config.OUTPUTS_DIR / "doc_number_index.json", encoding="utf-8") as f:
        doc_number_index = json.load(f)["usable"]
    context_id_to_doc_number = {v: k for k, v in doc_number_index.items()}

    dup_ids = build_dup_id_set(parsed_corpus)

    n_empty_text = 0
    n_merged_khoan = 0
    n_deduped_skips = 0
    n_underfilled = 0
    n_no_title = 0
    n_context_total = 0
    n_dup_id_used = 0
    n_with_ref = 0
    context_counts: dict[int, int] = {}
    total_syls: list[int] = []
    per_context_syls: list[int] = []
    packages = []

    with open(rerank_path, encoding="utf-8") as f_in:
        for line in f_in:
            r = json.loads(line)
            qid = r["qid"]
            if qid not in qa_src:
                raise SystemExit(
                    f"qid {qid} co trong ro nhung KHONG co trong nguon cau hoi "
                    f"'{args.questions}'. Ro va nguon cau hoi lech nhau — kiem "
                    f"--questions va --rerank-path co cung tap hay khong."
                )
            question = qa_src[qid]["question"]
            reference_answer = qa_src[qid].get("answer")
            if args.questions == "train":
                n_with_ref += 1 if (reference_answer or "").strip() else 0

            ranked = []
            for u in blended_rank(r["ranked_units"], args.alpha):
                ranked.append({
                    "unit_id": u["unit_id"],
                    "context_id": u["context_id"],
                    "text": get_unit_text(parsed_corpus, u["unit_id"]),
                    "score": u["score"],
                    "unit_type": infer_unit_type(u["unit_id"], u["context_id"]),
                })

            context_items = []
            seen_token_sets: list[set[str]] = []
            total_syl = 0
            for u in ranked:
                if len(context_items) >= args.max_contexts:
                    break
                if total_syl >= TARGET_TOTAL_SYLLABLES and len(context_items) >= 1:
                    break

                text, body_only, article_title, merged = decide_text(
                    u, parsed_corpus, args.per_item
                )
                if not text.strip():
                    n_empty_text += 1
                    continue

                # Dedup tren THAN, khong gom tieu de (diem 5 docstring module)
                tokens = normalize_tokens(body_only)
                if any(jaccard(tokens, seen) > JACCARD_DEDUP_THRESHOLD for seen in seen_token_sets):
                    n_deduped_skips += 1
                    continue

                n = syl_count(text)  # dem SAU khi dinh dang (diem 4)
                if context_items and total_syl + n > LEN_CAP_SYLLABLES:
                    break

                if merged:
                    n_merged_khoan += 1
                if not article_title:
                    n_no_title += 1
                if u["unit_id"] in dup_ids:
                    n_dup_id_used += 1
                n_context_total += 1
                per_context_syls.append(n)

                doc = parsed_corpus.get(u["context_id"], {})
                context_items.append(build_context_item(
                    context_id=u["context_id"],
                    unit_id=u["unit_id"],
                    unit_type=u["unit_type"],
                    text=text,
                    source_name=doc.get("name", ""),
                    source_link=doc.get("link", ""),
                    document_number=context_id_to_doc_number.get(u["context_id"], ""),
                    retrieval_score=u["score"],
                    article_title=article_title,
                ))
                seen_token_sets.append(tokens)
                total_syl += n

            if len(context_items) < 1:
                n_underfilled += 1
            context_counts[len(context_items)] = context_counts.get(len(context_items), 0) + 1
            total_syls.append(total_syl)
            packages.append(build_qa_package(qid, question, context_items, reference_answer))

    n_written = write_qa_packages_json(packages, out_path)

    total_syls.sort()
    per_context_syls.sort()
    n = len(total_syls)
    median = total_syls[n // 2]
    print(f"\nDa ghi {n_written} QAPackage -> {out_path}", flush=True)
    print(f"Variant dung: don={SINGLE_VARIANT} dai={RANGE_VARIANT} "
          f"cap_tieu_de={MAX_TITLE_SYLLABLES}", flush=True)
    print(f"alpha={args.alpha} MAX_CONTEXTS={args.max_contexts} "
          f"PER_ITEM={args.per_item} TARGET={TARGET_TOTAL_SYLLABLES}", flush=True)
    print(f"\nSo context rong (bi bo qua)              = {n_empty_text}", flush=True)
    print(f"So Khoan da gop lien ke                  = {n_merged_khoan}", flush=True)
    print(f"So ung vien bi bo vi TRUNG (Jaccard>{JACCARD_DEDUP_THRESHOLD}) = {n_deduped_skips}", flush=True)
    print(f"So cau KHONG co context nao (underfilled) = {n_underfilled}", flush=True)
    print(f"Phan bo so context/cau: {dict(sorted(context_counts.items()))}", flush=True)
    print(f"\nDo dai TONG (am tiet): p10={total_syls[n//10]} median={median} "
          f"p90={total_syls[9*n//10]} mean={sum(total_syls)/n:.0f}", flush=True)
    if per_context_syls:
        m = len(per_context_syls)
        print(f"Do dai MOI DOAN (am tiet): p10={per_context_syls[m//10]} "
              f"median={per_context_syls[m//2]} p90={per_context_syls[9*m//10]}", flush=True)
    print(f"  <LEN_FLOOR(250)={sum(1 for s in total_syls if s<250)/n:.1%}  "
          f">LEN_CAP(1200)={sum(1 for s in total_syls if s>1200)/n:.1%}", flush=True)
    print(f"Kich thuoc file: {out_path.stat().st_size / 1024:.1f} KB", flush=True)

    # --- Kiem sau build (plan v2 §4 P2) — BAT BUOC doc truoc khi gui C --------
    print("\n=== KIEM SAU BUILD ===", flush=True)

    pct_no_title = n_no_title / n_context_total if n_context_total else 1.0
    ok_title = pct_no_title < 0.10
    print(f"[{'OK ' if ok_title else 'CANH BAO'}] context khong co article_title: "
          f"{n_no_title}/{n_context_total} = {pct_no_title:.1%}", flush=True)
    if not ok_title:
        print("      Doi chieu: corpus that chi 1,1% Dieu khong co tieu de va 5,6%", flush=True)
        print("      co text==tieu_de -> ky vong ~7%. Cao hon nhieu = bug tra Dieu.", flush=True)

    # Goi train dung de DO OFFLINE — kernel tu cham bang score_if_available().
    # Ham do tra ve "not_available_incomplete_reference" va KHONG cham gi ca neu
    # chi mot dong thieu reference_answer. Bat o day, dung de phat hien sau 20
    # phut GPU ma khong co diem nao.
    if args.questions == "train":
        ok_ref = n_with_ref == n
        print(f"[{'OK ' if ok_ref else 'CHAN   '}] cau co reference_answer: "
              f"{n_with_ref}/{n}", flush=True)
        if not ok_ref:
            print("      score_if_available() doi DU 100%. Thieu mot dong la no tra", flush=True)
            print("      'not_available_incomplete_reference' va bo cham toan bo.", flush=True)
            print("      Loc bo cac qid thieu dap an khoi ro truoc khi chay kernel.", flush=True)

    pct_dup = n_dup_id_used / n_context_total if n_context_total else 0.0
    print(f"[INFO   ] context mang unit_id TRUNG: {n_dup_id_used}/{n_context_total} "
          f"= {pct_dup:.1%}", flush=True)
    print("      E1: 100% ban sao nam cung mot Dieu, nen sai lech toi da la lay nham", flush=True)
    print("      Khoan anh em trong DUNG Dieu. Duoi ~5% thi bo qua (E2 cuoi hang doi).", flush=True)

    ref_median = args.expected_median if args.expected_median is not None else V6_MEDIAN_SYLLABLES
    ref_label = "muc mong doi" if args.expected_median is not None else f"V6 ({V6_MEDIAN_SYLLABLES})"
    shift = median - ref_median
    ok_shift = abs(shift) <= MAX_ALLOWED_MEDIAN_SHIFT
    print(f"[{'OK ' if ok_shift else 'CANH BAO'}] median lech so {ref_label}: "
          f"{shift:+d} am tiet (cho phep +-{MAX_ALLOWED_MEDIAN_SHIFT})", flush=True)
    if not ok_shift:
        print("      Lech qua nhieu = bug dem am tiet hoac gop qua tay. Doc 5 cau truoc khi gui C.", flush=True)
    if args.expected_median is None:
        print("      Luu y: MAX_CONTEXTS giam 6->3 nen median GIAM la binh thuong.", flush=True)
        print("      Doi cau hinh (k=2) thi truyen --expected-median cho dung moc.", flush=True)
    else:
        print("      DO DAI TONG moi la bien that su quyet dinh, khong phai --per-item.", flush=True)
        print("      Do 19/09: k=1/k=2/k=3 deu dat dinh quanh tong 450-500 am tiet.", flush=True)
        print("      Median lech xa 450 thi chinh --per-item roi build lai.", flush=True)

    print("\n=== 3 VI DU KHOI 2 — DOC MAT, TIM TIEU DE LAP ===", flush=True)
    shown = 0
    for pkg in packages:
        for c in pkg["contexts"]:
            if not c["article_title"]:
                continue
            lines = c["text"].split("\n")
            dup = len(lines) >= 2 and lines[0].strip() == lines[1].strip()
            print(f"\n  qid={pkg['id']} {c['article']} {c['clause']}"
                  f"{'  <<< TIEU DE LAP!' if dup else ''}", flush=True)
            print(f"    article_title: {c['article_title'][:70]}", flush=True)
            for ln in lines[:3]:
                print(f"    | {ln[:90]}", flush=True)
            shown += 1
            break
        if shown >= 3:
            break

    print("\nGhi ket qua vao plan §7 TRUOC khi gui C. Nguong: LB >= nen + 0,006.", flush=True)
    print(f"Ghi kem md5 rerank ({rerank_md5}) va cau hinh alpha/k/per_item.", flush=True)


if __name__ == "__main__":
    main()
