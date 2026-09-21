"""T10 — GIA CUA VIEC VIET KHOI 3 KHI DAP AN CHUAN KHONG CO KHOI 3.

Phat hien dan den phep do nay:
  - 1000 cau dang dung de danh gia tach khoi duoc 100%
  - 6000 cau con lai chi 53,4%   -> tap danh gia DA BI LOC SAN
  - trong so cau khong tach duoc, 32,8% co doan cuoi VAN LA TRICH LUAT
    -> dap an chuan dung ngay sau Khoi 2, KHONG co ket luan
  - nhung he thong ta LUON sinh Khoi 3

Cau hoi: viet mot ket luan khong ai yeu cau thi mat bao nhieu METEOR?

Cach do (kieu "generator hoan hao", da dung nhieu lan trong du an):
  A = dap an chuan nguyen ven                      -> METEOR = 1,0000
  B = dap an chuan + MOT KET LUAN THAT do model sinh
Chenh lech A-B la gia thuan cua viec viet thua, moi thu khac giu hoan hao.

Ket luan dem vao lay tu chinh dau ra V9 tren train, nen do dai va van phong
dung nhu thu he thong that sinh ra - khong phai van ban bia.

Chay: PYTHONIOENCODING=utf-8 python pipeline/eval_khoi3_thua.py
"""
import json
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import config, io_utils
from nltk.translate.meteor_score import meteor_score

SEED = 42
N_BOOTSTRAP = 2000
DETAILS = Path("C:/Users/GIGA/AppData/Local/Temp/claude/"
               "c--Users-GIGA-VisualStudio2022Projects-job-hien-tai-dsc2026/"
               "a5d87954-e2b4-4e8f-ab41-daa8eff385fc/scratchpad/"
               "new_v9/model_production_outputs/kaggle-production/details.jsonl")

MARKERS = ("Như vậy", "Theo đó", "Do đó", "Vì vậy", "Từ đó", "Như đã phân tích", "Tóm lại")
_RE = re.compile(r"(?:^|\n)\s*(" + "|".join(re.escape(m) for m in MARKERS) + r")\b")
_LAWISH = re.compile(r"^\s*([0-9]+[.)]|[-–•]|Điều|Khoản|Chương|[a-z][.)])")


def met(ref, hyp):
    if not hyp.strip():
        return 0.0
    return float(meteor_score([ref.split()], hyp.split(), alpha=0.9, beta=3.0, gamma=0.5))


def boot(diffs):
    rng = random.Random(SEED)
    n = len(diffs)
    m = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOTSTRAP))
    return statistics.fmean(diffs), m[int(0.025 * N_BOOTSTRAP)], m[int(0.975 * N_BOOTSTRAP)]


train = io_utils.load_train()
ev = set(json.load(open(config.OUTPUTS_DIR / "eval_1000_qids.json", encoding="utf-8")))

# ket luan THAT do V9 sinh ra
concls = []
for line in open(DETAILS, encoding="utf-8"):
    c = (json.loads(line)["generation"].get("conclusion") or "").strip()
    if c:
        concls.append(c)
cl = sorted(len(c.split()) for c in concls)
print(f"kết luận thật do V9 sinh: {len(concls)} bản, trung vị {cl[len(cl)//2]} âm tiết, "
      f"p90 {cl[9*len(cl)//10]}\n")

# nhom cau: dap an chuan KHONG co Khoi 3 (doan cuoi van la trich luat)
no_concl, has_concl = [], []
for q, r in train.items():
    if q in ev or not r.get("answer"):
        continue
    t = r["answer"].replace("\r\n", "\n").strip()
    if "\n" not in t:
        continue
    paras = [p.strip() for p in t.split("\n") if p.strip()]
    if _RE.search(t, t.find("\n")):
        has_concl.append((q, t))
    elif _LAWISH.match(paras[-1]):
        no_concl.append((q, t))

print(f"nhóm A — đáp án chuẩn KHÔNG có Khối 3 : {len(no_concl)} câu")
print(f"nhóm B — đáp án chuẩn CÓ Khối 3       : {len(has_concl)} câu\n")

rng = random.Random(SEED)


def run(group, label):
    diffs, base = [], []
    for q, gold in group:
        extra = concls[rng.randrange(len(concls))]
        hyp = gold + "\n" + extra
        s = met(gold, hyp)
        base.append(s)
        diffs.append(s - 1.0)
    d, lo, hi = boot(diffs)
    gl = sorted(len(g.split()) for _, g in group)
    print(f"{label}")
    print(f"  n = {len(group)}   độ dài đáp án chuẩn trung vị {gl[len(gl)//2]} âm tiết")
    print(f"  METEOR khi thêm một kết luận thừa : {statistics.fmean(base):.4f}")
    print(f"  mất so với đáp án hoàn hảo        : {d:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]\n")
    return d


da = run(no_concl, "=== NHÓM A — đáp án chuẩn KHÔNG có Khối 3 (ta vẫn viết) ===")
db = run(has_concl, "=== NHÓM B — đáp án chuẩn CÓ Khối 3 (đối chứng: thêm MỘT kết luận NỮA) ===")

print("=== QUY RA TOÀN TẬP ===")
tot = len(no_concl) + len(has_concl)
share = len(no_concl) / tot
print(f"  tỷ lệ câu không cần Khối 3 (trong 6000 câu ngoài tập đánh giá): {share:.1%}")
print(f"  nếu private có cùng tỷ lệ, tổn thất kỳ vọng ≈ {share * abs(da):.4f} METEOR")
print(f"\n  đối chiếu: khoảng cách train → private chưa giải thích được = 0,0157")
print(f"             hiệu số V9 − V8 cả đợt thí nghiệm giành được    = 0,0337")
print("\n  LƯU Ý: đây là cận TRÊN của tổn thất. Nó giả định mọi thứ khác hoàn hảo;")
print("  trong thực tế Khối 1 và Khối 2 của ta cũng đã lệch nên phần trăm mất thêm")
print("  do Khối 3 thừa sẽ nhỏ hơn. Và nhóm A đo bằng quy tắc 'đoạn cuối là trích luật',")
print("  một xấp xỉ — không phải nhãn thật.")
