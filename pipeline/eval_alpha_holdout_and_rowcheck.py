"""Ba phep kiem truoc khi build:

1. ALPHA co bi KHOP QUA khong — kiem cheo hai nua (chon alpha tren nua A, cham
   tren nua B, roi dao lai). Alpha la THAM SO CHINH DUOC nen chinh sai duoc;
   alpha=1,0 la mot GOC nen khong chinh sai duoc. Do la bat doi xung that.
   Task 1 dinh dung loi nay: tham so `n` dao dau giua tap kiem tra va bang that.

   LUU Y: ro hybrid+rerank chi ton tai cho mau 1909 (kernel T3b chi chay 1.000
   cau do). Mau 1.000 MOI doi mot luot Kaggle 20 phut. Kiem cheo hai nua la
   duong sach khong ton GPU.

2. BOOTSTRAP tai k=2 — con so THAT SU quyet dinh (san xuat giao 2 doan) ma chua
   ai bootstrap. Bang truoc chi co CI cho top-1 va top-3.

3. row=-1 GIAO voi danh sach id trung — duong lui tra bang unit_id co sach khong.
   `get_unit_text` tra ban GAP DAU TIEN, ma 7.631 unit_id bi trung: 100% trung
   trong cung mot Dieu nhung 95,6% KHAC CHU (Jaccard median 0,107). Nen ung vien
   vua co row=-1 vua co unit_id trung se lay DUNG DIEU nhung co the SAI KHOAN.
   Gan 0 -> duong lui sach. Dang ke -> E2 nhay len thanh viec CHAN.

Chay: PYTHONIOENCODING=utf-8 python pipeline/eval_alpha_holdout_and_rowcheck.py
"""
import json
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import config, io_utils
from nltk.translate.meteor_score import meteor_score

L3_DIR = config.OUTPUTS_DIR / "layer3"
ARM = "title"
RRF_K = 60
BUDGET = 400          # giu nhu bang truoc de so duoc; ngan sach toi uu do rieng
ALPHAS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
K_PROD = 2            # san xuat giao 2 doan
N_BOOTSTRAP = 2000
SEED = 42

MARKERS_VN = ("Như vậy", "Theo đó", "Do đó", "Vì vậy",
              "Từ đó", "Như đã phân tích", "Tóm lại")
_RE = re.compile(r"(?:^|\n)\s*(" + "|".join(re.escape(m) for m in MARKERS_VN) + r")\b")


def split_blocks(a):
    t = (a or "").replace("\r\n", "\n").strip()
    if not t:
        return "", "", ""
    nl = t.find("\n")
    if nl == -1:
        return "", "", ""
    lead = t[:nl].strip()
    rs = nl
    while rs < len(t) and t[rs] in "\n \t":
        rs += 1
    ms = list(_RE.finditer(t, rs))
    if not ms:
        return "", "", ""
    cut = ms[-1].start(1)
    q, c = t[rs:cut].strip(), t[cut:].strip()
    if not (lead and q and c):
        return "", "", ""
    return lead, q, c


def syl(t):
    return len(t.split())


def met(ref, hyp):
    if not hyp.strip():
        return 0.0
    return float(meteor_score([ref.split()], hyp.split(), alpha=0.9, beta=3.0, gamma=0.5))


def expand(kl, idx, budget):
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


print("Load...", flush=True)
t0 = time.time()
parsed = io_utils.load_parsed_corpus()
train = io_utils.load_train()
row_map, row_uid = [], []
with open(config.DATA_DIR / "parsed_corpus.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        cid = r["context_id"]
        if r["parse_status"] == "fallback":
            if r["dieu"]:
                d0 = r["dieu"][0]
                row_map.append((cid, 0, -1))
                row_uid.append(d0["dieu_id"] or cid)
            continue
        for di, dieu in enumerate(r["dieu"]):
            if dieu["khoan"]:
                for ki, k in enumerate(dieu["khoan"]):
                    row_map.append((cid, di, ki))
                    row_uid.append(k["khoan_id"])
            else:
                row_map.append((cid, di, -1))
                row_uid.append(dieu["dieu_id"])
uid_map = {}
for _pos, _u in enumerate(row_uid):
    uid_map.setdefault(_u, _pos)
DUP_IDS = {u for u, c in Counter(row_uid).items() if c > 1}
print(f"  {time.time()-t0:.0f}s | {len(row_uid)} hang, {len(DUP_IDS)} ma trung\n", flush=True)


def resolve(u):
    r = u.get("row", -1)
    if r is None or r < 0:
        r = uid_map.get(u.get("unit_id"), -1)
    return r


_cache = {}


def text_of(row):
    if row in _cache:
        return _cache[row]
    out = ""
    if 0 <= row < len(row_map):
        cid, di, ki = row_map[row]
        doc = parsed.get(cid)
        if doc and di < len(doc.get("dieu", [])):
            dieu = doc["dieu"][di]
            kl = dieu.get("khoan", [])
            out = dieu.get("text", "") if (ki < 0 or not kl) else "\n\n".join(
                k.get("text", "") for k in expand(kl, ki, BUDGET) if k.get("text", "").strip())
    _cache[row] = out
    return out


# ---------------------------------------------------------------------------
# 3. row=-1 GIAO voi id trung  (lam truoc, re nhat, va no CHAN viec build)
# ---------------------------------------------------------------------------
print("=" * 68)
print("3. DUONG LUI tra bang unit_id co sach khong?")
print("=" * 68)
for arm in ("bare", "title"):
    p = L3_DIR / f"L3_rerank_train_{arm}.jsonl"
    if not p.exists():
        continue
    n_tot = n_neg = n_neg_dup = n_dup_any = 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            for u in json.loads(line)["ranked_units"]:
                n_tot += 1
                is_dup = u.get("unit_id") in DUP_IDS
                if is_dup:
                    n_dup_any += 1
                if u.get("row", -1) is None or u.get("row", -1) < 0:
                    n_neg += 1
                    if is_dup:
                        n_neg_dup += 1
    print(f"\n  {arm}: {n_tot:,} ung vien")
    print(f"    row = -1                      : {n_neg:,} = {n_neg/n_tot:.1%}")
    print(f"    unit_id trung (bat ke row)    : {n_dup_any:,} = {n_dup_any/n_tot:.1%}")
    print(f"    VUA row=-1 VUA id trung       : {n_neg_dup:,} = {n_neg_dup/n_tot:.2%}"
          f"  <-- duong lui co the sai KHOAN")
    print(f"      (trong so row=-1: {n_neg_dup/max(n_neg,1):.2%})")

# ---------------------------------------------------------------------------
# chuan bi cases
# ---------------------------------------------------------------------------
cases = []
with open(L3_DIR / f"L3_rerank_train_{ARM}.jsonl", encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        lead, quote, concl = split_blocks(train[o["qid"]]["answer"])
        if not quote or len(quote.split()) < 10:
            continue
        items = [(resolve(u), u.get("dense_rank", 1 << 20), r)
                 for r, u in enumerate(o["ranked_units"])]
        cases.append((o["qid"], lead, quote, concl, items))
print(f"\nn={len(cases)}  nhanh={ARM}  k={K_PROD}  ngan sach={BUDGET}\n", flush=True)


def score_case(case, a, k):
    _qid, lead, quote, concl, items = case
    o = sorted(items, key=lambda it: -(a / (RRF_K + it[1]) + (1 - a) / (RRF_K + it[2])))
    parts = [t for t in (text_of(it[0]) for it in o[:k]) if t.strip()]
    hyp = "\n\n".join(parts)
    return met(f"{lead}\n{quote}\n{concl}", f"{lead}\n{hyp}\n{concl}")


print("=" * 68)
print(f"2. BOOTSTRAP tai k={K_PROD} — con so that su quyet dinh")
print("=" * 68)
t0 = time.time()
by_alpha = {a: [score_case(c, a, K_PROD) for c in cases] for a in ALPHAS}
print(f"{'alpha':>7}{'METEOR ca bai':>16}")
for a in ALPHAS:
    print(f"{a:>7.1f}{statistics.fmean(by_alpha[a]):>16.4f}")
d, lo, hi = paired_bootstrap(by_alpha[1.0], by_alpha[0.8])
flag = " *" if lo > 0 else (" !" if hi < 0 else "  (trong nhieu)")
print(f"\n  alpha=0,8 vs alpha=1,0 tai k={K_PROD}: {d:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]{flag}")
b0 = sum(1 for x, y in zip(by_alpha[1.0], by_alpha[0.8]) if y > x + 1e-9)
w0 = sum(1 for x, y in zip(by_alpha[1.0], by_alpha[0.8]) if y < x - 1e-9)
print(f"  tot hon {b0} / te hon {w0} / khong doi {len(cases)-b0-w0}")
print(f"  ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------------------
# 1. kiem cheo hai nua
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("1. ALPHA co KHOP QUA khong — kiem cheo hai nua")
print("=" * 68)
idx = list(range(len(cases)))
random.Random(7).shuffle(idx)
half = len(idx) // 2
folds = [(idx[:half], idx[half:]), (idx[half:], idx[:half])]
for i, (tr, te) in enumerate(folds, 1):
    best_a = max(ALPHAS, key=lambda a: statistics.fmean([by_alpha[a][j] for j in tr]))
    on_te = statistics.fmean([by_alpha[best_a][j] for j in te])
    one_te = statistics.fmean([by_alpha[1.0][j] for j in te])
    a_tr = [by_alpha[1.0][j] for j in te]
    b_tr = [by_alpha[best_a][j] for j in te]
    d, lo, hi = paired_bootstrap(a_tr, b_tr)
    print(f"\n  Nua {i}: chon tren {len(tr)} cau -> alpha={best_a:.1f}")
    print(f"    cham tren {len(te)} cau GIU RIENG: alpha={best_a:.1f} {on_te:.4f}  "
          f"vs alpha=1,0 {one_te:.4f}")
    print(f"    chenh {d:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]"
          f"{' *' if lo > 0 else (' !' if hi < 0 else '  (trong nhieu)')}")

print("\n--- DOC SO ---")
print("Neu ca hai nua deu chon alpha < 1,0 VA thang tren nua giu rieng -> khop qua")
print("khong phai van de, lay alpha. Neu dao dau giua hai nua -> lay alpha=1,0.")
