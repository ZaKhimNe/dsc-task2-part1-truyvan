"""T1 - Gold Dieu dang o HANG MAY trong 50 ung vien?

CUA QUYET DINH: hang 4-50 -> loi o khau CHAM (reranker, T3).
                ngoai 50  -> loi o khau SINH UNG VIEN (T4 / cho cat / nhung).
Hai nhanh can hai cach sua khac han. Task 1 mat 11 ngay vi doc so o nhanh sai.

NGUONG DAT TRUOC (da lap lo o giua):
  ngoai 50 >= 50%   -> nut that o sinh ung vien, uu tien T4
  ngoai 50  30-50%  -> VAN T3 truoc (T3 re hon nhieu, va no tra loi luon mot
                       phan cau hoi cua T4)
  ngoai 50  < 30%   -> nut that o xep hang, uu tien T3

CACH DO: nhan theo chuoi (khong dung ma, mien nhiem bug trung ma). Voi moi cau,
tim unit trong top-50 phu >= 70% chu cua Khoi 2. Hang = vi tri dau tien.
Khong unit nao dat -> "ngoai 50".

GIOI HAN PHAI GHI RO: ro da xuat chi co top-50, nen "ngoai 50" chi biet la NGOAI,
khong biet xa bao nhieu. Phan biet hang 51 voi hang 5.000 can chay lai search voi
K lon hon — chi lam neu nhom nay du lon de dang.

Chay: PYTHONIOENCODING=utf-8 python pipeline/eval_gold_dieu_rank.py
"""
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import config, io_utils

E4_DIR = config.OUTPUTS_DIR / "e4"
N_SAMPLE = 1000
SEED = 2026
COV_THRESHOLD = 0.70

MARKERS_VN = ("Như vậy", "Theo đó", "Do đó", "Vì vậy",
              "Từ đó", "Như đã phân tích", "Tóm lại")
_RE = re.compile(r"(?:^|\n)\s*(" + "|".join(re.escape(m) for m in MARKERS_VN) + r")\b")


def split_quote(a):
    t = (a or "").replace("\r\n", "\n").strip()
    if not t:
        return ""
    nl = t.find("\n")
    if nl == -1:
        return ""
    rs = nl
    while rs < len(t) and t[rs] in "\n \t":
        rs += 1
    ms = list(_RE.finditer(t, rs))
    if not ms:
        return ""
    q = t[rs:ms[-1].start(1)].strip()
    return q if (t[:nl].strip() and q) else ""


def cov(gt, text):
    if not gt:
        return 0.0
    s = set(text.lower().split())
    return sum(1 for x in gt if x in s) / len(gt)


print("Load parsed_corpus + train.json...", flush=True)
t0 = time.time()
parsed = io_utils.load_parsed_corpus()
train = io_utils.load_train()
row_map = []
with open(config.DATA_DIR / "parsed_corpus.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        cid = r["context_id"]
        if r["parse_status"] == "fallback":
            if r["dieu"]:
                row_map.append((cid, 0, -1))
            continue
        for di, dieu in enumerate(r["dieu"]):
            if dieu["khoan"]:
                for ki in range(len(dieu["khoan"])):
                    row_map.append((cid, di, ki))
            else:
                row_map.append((cid, di, -1))
print(f"  {len(row_map)} hang, {time.time()-t0:.0f}s\n", flush=True)


def unit_of(row):
    cid, di, ki = row_map[row]
    doc = parsed.get(cid)
    if not doc or di >= len(doc.get("dieu", [])):
        return None, None
    dieu = doc["dieu"][di]
    kl = dieu.get("khoan", [])
    unit = kl[ki] if (0 <= ki < len(kl)) else dieu
    return dieu, unit


BUCKETS = [("hang 1", 1, 1), ("hang 2-3", 2, 3), ("hang 4-10", 4, 10),
           ("hang 11-50", 11, 50)]

for arm in ("bare", "title"):
    path = E4_DIR / f"top50_{arm}_qa_train.jsonl"
    basket = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            basket[o["qid"]] = o["rows"]

    qids = [q for q in sorted(basket) if train.get(q, {}).get("answer")]
    random.Random(SEED).shuffle(qids)

    ranks = []      # hang (1-based) hoac None neu ngoai 50
    done = 0
    t0 = time.time()
    for qid in qids:
        if done >= N_SAMPLE:
            break
        quote = split_quote(train[qid]["answer"])
        if not quote:
            continue
        gt = quote.lower().split()
        if len(gt) < 10:
            continue
        done += 1
        found = None
        for pos, r in enumerate(basket[qid]):
            dieu, unit = unit_of(r)
            if unit is None:
                continue
            # cap KHOAN truoc; neu khong dat thi thu ca DIEU (gold hay trai nhieu Khoan)
            if cov(gt, unit.get("text", "")) >= COV_THRESHOLD or \
               cov(gt, dieu.get("text", "")) >= COV_THRESHOLD:
                found = pos + 1
                break
        ranks.append(found)
        if done % 250 == 0:
            print(f"  [{arm}] {done}/{N_SAMPLE}  {time.time()-t0:.0f}s", flush=True)

    n = len(ranks)
    inside = [r for r in ranks if r is not None]
    n_out = n - len(inside)

    print(f"\n=== NHANH {arm.upper()} — n={n}, nguong phu {COV_THRESHOLD:.0%} ===")
    cum = 0
    for name, lo, hi in BUCKETS:
        c = sum(1 for r in inside if lo <= r <= hi)
        cum += c
        print(f"  {name:<12} {c:>5} = {c/n:>6.1%}   (cong don {cum/n:>6.1%})")
    print(f"  {'NGOAI 50':<12} {n_out:>5} = {n_out/n:>6.1%}")
    if inside:
        inside_sorted = sorted(inside)
        print(f"  hang median {inside_sorted[len(inside_sorted)//2]}  "
              f"mean {statistics.fmean(inside):.1f}  "
              f"p90 {inside_sorted[9*len(inside_sorted)//10]}")

    # nhom TRUOT = khong o hang 1 (ke ca ngoai 50)
    n_miss = n - sum(1 for r in inside if r == 1)
    n_out_share = n_out / n_miss if n_miss else 0.0
    print(f"\n  Trong {n_miss} cau KHONG dat hang 1:")
    print(f"    ngoai 50            : {n_out} = {n_out_share:.1%}")
    print(f"    trong 50 nhung sai hang: {n_miss-n_out} = {1-n_out_share:.1%}")
    if n_out_share >= 0.50:
        verdict = "NUT THAT O SINH UNG VIEN -> uu tien T4"
    elif n_out_share >= 0.30:
        verdict = "VUNG GIUA 30-50% -> VAN T3 TRUOC (theo nguong da lap lo)"
    else:
        verdict = "NUT THAT O XEP HANG -> uu tien T3"
    print(f"    => {verdict}\n", flush=True)

print("--- GIOI HAN ---")
print("Ro chi co top-50 nen 'ngoai 50' khong biet xa bao nhieu.")
print("Do tren ro DENSE THUAN cua E4, chua qua hybrid/rerank.")
print("T2 da do rieng: 21,89% gold nam sau cho cat 50.000 ky tu -> BM25 mu,")
print("nhung dense khong dinh, nen con so o day KHONG bi cho cat lam nhiem.")
