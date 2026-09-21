"""TRAN CUA KHAU XEP HANG: neu xep hang HOAN HAO trong chinh ro 50 dang co
thi duoc bao nhieu?

Vi sao do cai nay truoc khi chon huong dau tu (quy tac 9 cua Task 1 - do tran
TRUOC khi chay):

  - T1 da do: gold Dieu nam ngoai top-50 chi 9,6-13,7% -> ung vien KHONG phai nut that
  - T3 da do: reranker co tran ~60%, va tren nhanh title no LAM HAI (64,8 -> 60,7)
  -> nut that nam o XEP HANG. Nhung "nut that" chua noi duoc CON BAO NHIEU de an.

Ba moc, cung mot harness, cung ngan sach, cung k:

  san xuat     : xep bang RRF trong so alpha=0,7, lay 2 dau
  tran top-10  : chon 2 TOT NHAT trong 10 dau cua san xuat  -> loi neu chi sua PHAN DAU
  tran top-50  : chon 2 TOT NHAT trong ca 50               -> loi neu sua het xep hang

Chon "tot nhat" = tham lam theo METEOR cua ca bai (Khoi 1+3 lay tu gold).
Tham lam la CAN DUOI cua tran that; chenh lech voi vet can 1225 cap la nho vi
METEOR gan don dieu theo do phu.

Chay: PYTHONIOENCODING=utf-8 python pipeline/eval_tran_xep_hang.py
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

L3_DIR = config.OUTPUTS_DIR / "layer3"
ARM = "title"
ALPHA = 0.7          # cau hinh san xuat V9
RRF_K = 60
BUDGET = 200         # cau hinh san xuat V9
K = 2                # cau hinh san xuat V9
POOLS = [10, 50]
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
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(N_BOOTSTRAP))
    return delta, means[int(0.025 * N_BOOTSTRAP)], means[int(0.975 * N_BOOTSTRAP)]


print("Load...", flush=True)
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
uid_map = {}
for _pos, (_cid, _di, _ki) in enumerate(row_map):
    _doc = parsed.get(_cid)
    if not _doc or _di >= len(_doc.get("dieu", [])):
        continue
    _dieu = _doc["dieu"][_di]
    _kl = _dieu.get("khoan", [])
    _uid = _kl[_ki]["khoan_id"] if (0 <= _ki < len(_kl)) else _dieu.get("dieu_id")
    uid_map.setdefault(_uid, _pos)
print(f"  {time.time()-t0:.0f}s", flush=True)


def resolve(u):
    r = u.get("row", -1)
    if r is None or r < 0:
        r = uid_map.get(u.get("unit_id"), -1)
    return r


def text_of(row, budget):
    if row < 0 or row >= len(row_map):
        return ""
    cid, di, ki = row_map[row]
    doc = parsed.get(cid)
    if not doc or di >= len(doc.get("dieu", [])):
        return ""
    dieu = doc["dieu"][di]
    kl = dieu.get("khoan", [])
    if ki < 0 or not kl:
        return dieu.get("text", "")
    return "\n\n".join(k.get("text", "") for k in expand(kl, ki, budget)
                       if k.get("text", "").strip())


cases = []
with open(L3_DIR / f"L3_rerank_train_{ARM}.jsonl", encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        lead, quote, concl = split_blocks(train[o["qid"]]["answer"])
        if not quote or len(quote.split()) < 10:
            continue
        items = [(resolve(u), u.get("dense_rank", 1 << 20), r)
                 for r, u in enumerate(o["ranked_units"])]
        items.sort(key=lambda it: -(ALPHA / (RRF_K + it[1]) + (1 - ALPHA) / (RRF_K + it[2])))
        cases.append((lead, quote, concl, items))

print(f"n={len(cases)}  nhanh={ARM}  alpha={ALPHA}  ngan sach={BUDGET}  k={K}")
print(f"harness=CA BAI (Khoi 1+3 tu gold)\n", flush=True)

prod, oracle = [], {p: [] for p in POOLS}
pos_hist = {p: [] for p in POOLS}
t0 = time.time()
for ci, (lead, quote, concl, items) in enumerate(cases):
    ref = f"{lead}\n{quote}\n{concl}"
    texts = [text_of(it[0], BUDGET) for it in items[:max(POOLS)]]

    def score(sel):
        body = "\n\n".join(texts[i] for i in sel if texts[i].strip())
        return met(ref, f"{lead}\n{body}\n{concl}")

    prod.append(score([0, 1]))

    for P in POOLS:
        pool = range(min(P, len(texts)))
        best1 = max(pool, key=lambda i: score([i]))
        rest = [i for i in pool if i != best1]
        best2 = max(rest, key=lambda i: score(sorted([best1, i]))) if rest else best1
        sel = sorted({best1, best2})
        oracle[P].append(score(sel))
        pos_hist[P].append(tuple(sel))

    if (ci + 1) % 100 == 0:
        print(f"  {ci+1}/{len(cases)}  {time.time()-t0:.0f}s", flush=True)

print(f"\n{'moc':<28}{'METEOR':>10}{'so voi san xuat':>28}")
print("-" * 66)
print(f"{'san xuat (alpha=0,7, k=2)':<28}{statistics.fmean(prod):>10.4f}{'':>28}")
for P in POOLS:
    d, lo, hi = paired_bootstrap(prod, oracle[P])
    print(f"{'tran: chon 2 tot nhat trong '+str(P):<28}"
          f"{statistics.fmean(oracle[P]):>10.4f}"
          f"{f'{d:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]':>28}")

print("\n=== Hai suat toi uu nam o hang nao (tinh tren thu tu san xuat) ===")
for P in POOLS:
    flat = [i for sel in pos_hist[P] for i in sel]
    n = len(flat)
    print(f"  pool {P}: hang 1-2 {sum(1 for i in flat if i < 2)/n:.1%}   "
          f"hang 3-5 {sum(1 for i in flat if 2 <= i < 5)/n:.1%}   "
          f"hang 6-10 {sum(1 for i in flat if 5 <= i < 10)/n:.1%}   "
          f"hang 11+ {sum(1 for i in flat if i >= 10)/n:.1%}")
    same = sum(1 for sel in pos_hist[P] if sel == (0, 1))
    print(f"           san xuat da chon DUNG ca hai suat: {same}/{len(pos_hist[P])} "
          f"({same/len(pos_hist[P]):.1%})")
