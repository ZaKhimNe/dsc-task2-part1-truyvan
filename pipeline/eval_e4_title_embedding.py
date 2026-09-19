"""E4 - Cham nhanh BARE vs TITLE tren ro dense that (Kaggle), bang nhan theo chuoi.

Input: outputs/e4/ (giai nen tu e4_results.zip cua kernel
       zakhim/dsc-legalqa-e4-title-embedding)

NGUONG DAT TRUOC (plan v2 §4 P3): recall@50 >= +2,0 diem VA can duoi CI > 0.

CACH TRA TEXT - quan trong: dung ROW INDEX, khong dung unit_id.
Bug retrieve.py:413 lam 7.631 ma tro sai doan (E1). Kernel da xuat `rows` chinh la
vi tri hang trong ma tran embedding, nen o day ta duyet lai corpus DUNG THU TU CELL 5
de dung bang row -> (context_id, chi so Dieu, chi so Khoan). Tra kieu nay khong the sai.

Do hai thuoc, deu tren cung mot ro:
  1. "voi toi duoc": co Dieu nao trong top-50 phu >= nguong chu cua Khoi 2 khong
     -> day la thuoc tuong duong recall@50 ma nguong plan noi toi
  2. METEOR cua thu SE GIAO DI (mo rong ngan sach 400, theo E6b)

Chay: PYTHONIOENCODING=utf-8 python pipeline/eval_e4_title_embedding.py
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
from nltk.translate.meteor_score import meteor_score

E4_DIR = config.OUTPUTS_DIR / "e4"
N_SAMPLE = 1000
SEED = 2026
BUDGET = 400
N_BOOTSTRAP = 2000

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


def syl(t):
    return len(t.split())


def cov(gt, text):
    if not gt:
        return 0.0
    s = set(text.lower().split())
    return sum(1 for x in gt if x in s) / len(gt)


def met(gold, hyp):
    if not hyp.strip():
        return 0.0
    return float(meteor_score([gold.split()], hyp.split(), alpha=0.9, beta=3.0, gamma=0.5))


def expand(kl, idx, budget):
    if not kl:
        return []
    lo = hi = idx
    total = syl(kl[idx].get("text", ""))
    while total < budget:
        ch, cl = hi + 1 < len(kl), lo - 1 >= 0
        if not ch and not cl:
            break
        if ch:
            hi += 1
            total += syl(kl[hi].get("text", ""))
        else:
            lo -= 1
            total += syl(kl[lo].get("text", ""))
    return kl[lo:hi + 1]


def paired_bootstrap(a, b):
    rng = random.Random(SEED)
    n = len(a)
    diffs = [y - x for x, y in zip(a, b)]
    delta = statistics.fmean(diffs)
    means = []
    for _ in range(N_BOOTSTRAP):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return delta, means[int(0.025 * N_BOOTSTRAP)], means[int(0.975 * N_BOOTSTRAP)]


print("Load parsed_corpus + train.json...", flush=True)
t0 = time.time()
parsed = io_utils.load_parsed_corpus()
train = io_utils.load_train()
print(f"  {time.time()-t0:.0f}s", flush=True)

# Dung bang row -> (context_id, chi so Dieu, chi so Khoan) bang cach duyet lai
# DUNG THU TU CELL 5 cua embed_corpus_title_notebook.py
print("Dung bang row -> vi tri...", flush=True)
t0 = time.time()
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
print(f"  {len(row_map)} hang, {time.time()-t0:.0f}s", flush=True)

with open(E4_DIR / "unit_index.tsv", encoding="utf-8") as f:
    n_tsv = sum(1 for _ in f) - 1
assert n_tsv == len(row_map), f"LECH: unit_index.tsv {n_tsv} vs duyet lai {len(row_map)}"
print(f"  khop unit_index.tsv ({n_tsv})", flush=True)


def deliverable(row, gt):
    cid, di, ki = row_map[row]
    doc = parsed.get(cid)
    if not doc or di >= len(doc.get("dieu", [])):
        return 0.0, ""
    dieu = doc["dieu"][di]
    kl = dieu.get("khoan", [])
    if ki < 0 or not kl:
        t = dieu.get("text", "")
        return cov(gt, t), t
    items = expand(kl, ki, BUDGET)
    t = "\n\n".join(k.get("text", "") for k in items if k.get("text", "").strip())
    return cov(gt, t), t


def load_arm(arm):
    d = {}
    with open(E4_DIR / f"top50_{arm}_qa_train.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            d[o["qid"]] = o["rows"]
    return d


bare = load_arm("bare")
title = load_arm("title")
print(f"bare {len(bare)} cau · title {len(title)} cau", flush=True)

qids = [q for q in sorted(bare) if q in title and train.get(q, {}).get("answer")]
random.Random(SEED).shuffle(qids)

rows_out = []
t0 = time.time()
for qid in qids:
    if len(rows_out) >= N_SAMPLE:
        break
    quote = split_quote(train[qid]["answer"])
    if not quote:
        continue
    gt = quote.lower().split()
    if len(gt) < 10:
        continue
    cb = mb = 0.0
    tb = ""
    for r in bare[qid]:
        c, t = deliverable(r, gt)
        if c > cb:
            cb, tb = c, t
    ct = 0.0
    tt = ""
    for r in title[qid]:
        c, t = deliverable(r, gt)
        if c > ct:
            ct, tt = c, t
    rows_out.append((cb, ct, met(quote, tb), met(quote, tt)))
    if len(rows_out) % 100 == 0:
        print(f"  {len(rows_out)}/{N_SAMPLE}  {time.time()-t0:.0f}s", flush=True)

n = len(rows_out)
cb = [r[0] for r in rows_out]
ct = [r[1] for r in rows_out]
mb = [r[2] for r in rows_out]
mt = [r[3] for r in rows_out]

print(f"\nn={n}\n")
print("=== 1. VOI TOI DUOC trong top-50 (tuong duong recall@50) ===")
print(f"{'nguong phu':>11} {'BARE':>8} {'TITLE':>8} {'chenh':>9}   {'CI 95%':>22}")
for thr in (0.70, 0.80, 0.90, 0.95):
    a = [1.0 if v >= thr else 0.0 for v in cb]
    b = [1.0 if v >= thr else 0.0 for v in ct]
    d, lo, hi = paired_bootstrap(a, b)
    flag = " *" if lo > 0 else (" !" if hi < 0 else "")
    print(f"{thr:>10.0%} {statistics.fmean(a):>8.1%} {statistics.fmean(b):>8.1%} "
          f"{d*100:>+8.2f}d   [{lo*100:+.2f}, {hi*100:+.2f}]{flag}")

print("\n=== 2. PHU CHU trung binh ===")
d, lo, hi = paired_bootstrap(cb, ct)
print(f"  BARE {statistics.fmean(cb):.4f}  TITLE {statistics.fmean(ct):.4f}  "
      f"chenh {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")

print(f"\n=== 3. METEOR cua thu SE GIAO DI (ngan sach {BUDGET}) ===")
d, lo, hi = paired_bootstrap(mb, mt)
print(f"  BARE {statistics.fmean(mb):.4f}  TITLE {statistics.fmean(mt):.4f}  "
      f"chenh {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
better = sum(1 for x, y in zip(mb, mt) if y > x + 1e-9)
worse = sum(1 for x, y in zip(mb, mt) if y < x - 1e-9)
print(f"  tot hon: {better}  te hon: {worse}  khong doi: {n-better-worse}")

print("\n(* = can duoi CI > 0 -> TITLE thang that; ! = can tren CI < 0 -> TITLE THUA that)")
print("NGUONG PLAN: recall@50 >= +2,0 diem VA CI duoi > 0")
