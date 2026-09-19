"""Quet trong so alpha giua thu tu DENSE va thu tu RERANKER. 0 GPU.

Y TUONG: ca hai bo xep hang cung roi ve ~60% hang 1 (bare 54,9->60,1; title
64,8->60,7). Chung ghi de len nhau chu khong bo tro. Nhung neu chung sai o
NHUNG CAU KHAC NHAU thi tron lai se cao hon ca hai dau.

  diem cuoi = alpha * (phan dense) + (1-alpha) * (phan reranker)

alpha=1 -> "khong rerank" ; alpha=0 -> "rerank thuan".

LUU Y KY THUAT: kernel xuat `dense_rank` (THU HANG) chu khong xuat diem dense tho.
Nen tron theo THU HANG, khong tron theo diem. That ra tot hon — thang cua cosine
va thang cua cross-encoder khong so duoc, con thu hang thi vo thang. Day chinh la
RRF co trong so, dung thu da bi bo o tang union BM25/dense.

Do hai kieu tron:
  RRF  : s = a/(K + r_dense) + (1-a)/(K + r_rerank)      K=60
  LINEAR: r = a*r_dense + (1-a)*r_rerank                  (thap hon = tot hon)

Va do o hai muc: METEOR chi KHOI 2, va METEOR CA BAI (Khoi 1 + 3 lay tu gold) —
vi hai harness nay tung cho ket luan nguoc nhau ve top-1 vs top-3.

CHOT ALPHA TREN MAU TRAIN, KHONG CHOT BANG LB. Task 1 dinh dung loi nay:
tham so `n` dao dau giua tap kiem tra va bang that.

Chay: PYTHONIOENCODING=utf-8 python pipeline/eval_alpha_blend.py
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
BUDGET = 400
RRF_K = 60
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_BOOTSTRAP = 2000
SEED = 42

MARKERS_VN = ("Như vậy", "Theo đó", "Do đó", "Vì vậy",
              "Từ đó", "Như đã phân tích", "Tóm lại")
_RE = re.compile(r"(?:^|\n)\s*(" + "|".join(re.escape(m) for m in MARKERS_VN) + r")\b")


def split_blocks(a):
    """(lead, quotation, conclusion) hoac ("","","")."""
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
print(f"  {time.time()-t0:.0f}s\n", flush=True)

_text_cache = {}


def text_of(row):
    if row in _text_cache:
        return _text_cache[row]
    out = ""
    if 0 <= row < len(row_map):
        cid, di, ki = row_map[row]
        doc = parsed.get(cid)
        if doc and di < len(doc.get("dieu", [])):
            dieu = doc["dieu"][di]
            kl = dieu.get("khoan", [])
            if ki < 0 or not kl:
                out = dieu.get("text", "")
            else:
                items = expand(kl, ki, BUDGET)
                out = "\n\n".join(k.get("text", "") for k in items if k.get("text", "").strip())
    _text_cache[row] = out
    return out


def resolve(u):
    r = u.get("row", -1)
    if r is None or r < 0:
        r = uid_map.get(u.get("unit_id"), -1)
    return r


ARM = "title"
units_by_q = {}
with open(L3_DIR / f"L3_rerank_train_{ARM}.jsonl", encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        units_by_q[o["qid"]] = o["ranked_units"]

# chuan bi: moi cau -> list (row, r_dense, r_rerank)
cases = []
for qid, units in units_by_q.items():
    lead, quote, concl = split_blocks(train[qid]["answer"])
    if not quote or len(quote.split()) < 10:
        continue
    items = []
    for r_rr, u in enumerate(units):   # file da sap theo diem reranker
        items.append((resolve(u), u.get("dense_rank", 1 << 20), r_rr))
    cases.append((qid, lead, quote, concl, items))
print(f"n={len(cases)}  nhanh={ARM}  (mau seed 1909)\n", flush=True)


def order_rrf(items, a):
    return sorted(items, key=lambda it: -(a / (RRF_K + it[1]) + (1 - a) / (RRF_K + it[2])))


def order_linear(items, a):
    return sorted(items, key=lambda it: a * it[1] + (1 - a) * it[2])


def build(items_sorted, k):
    parts = []
    for it in items_sorted[:k]:
        t = text_of(it[0])
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts)


print("=== QUET ALPHA — METEOR chi KHOI 2 ===")
print(f"{'alpha':>7}{'RRF top-1':>12}{'RRF top-3':>12}{'LIN top-1':>12}{'LIN top-3':>12}")
print("-" * 55)
best = {}
store = {}
t0 = time.time()
for a in ALPHAS:
    vals = {("rrf", 1): [], ("rrf", 3): [], ("lin", 1): [], ("lin", 3): []}
    for qid, lead, quote, concl, items in cases:
        o_rrf = order_rrf(items, a)
        o_lin = order_linear(items, a)
        for k in (1, 3):
            vals[("rrf", k)].append(met(quote, build(o_rrf, k)))
            vals[("lin", k)].append(met(quote, build(o_lin, k)))
    store[a] = vals
    print(f"{a:>7.1f}{statistics.fmean(vals[('rrf',1)]):>12.4f}"
          f"{statistics.fmean(vals[('rrf',3)]):>12.4f}"
          f"{statistics.fmean(vals[('lin',1)]):>12.4f}"
          f"{statistics.fmean(vals[('lin',3)]):>12.4f}", flush=True)

for key in (("rrf", 1), ("rrf", 3), ("lin", 1), ("lin", 3)):
    b = max(ALPHAS, key=lambda a: statistics.fmean(store[a][key]))
    best[key] = b
    print(f"\n{key[0].upper()} top-{key[1]}: dinh o alpha={b:.1f} = "
          f"{statistics.fmean(store[b][key]):.4f}   "
          f"(alpha=0 rerank thuan {statistics.fmean(store[0.0][key]):.4f}, "
          f"alpha=1 dense thuan {statistics.fmean(store[1.0][key]):.4f})")
    d, lo, hi = paired_bootstrap(store[1.0][key], store[b][key])
    print(f"   dinh vs dense thuan: {d:+.4f} CI95 [{lo:+.4f}, {hi:+.4f}]"
          f"{' *' if lo > 0 else ''}")

# --- CA BAI: Khoi 1 va Khoi 3 lay tu gold ---
print(f"\n\n=== TOP-1 vs TOP-3, harness CA BAI (Khoi 1+3 tu gold) ===")
print("Bang truoc cham RIENG Khoi 2. Con so cu (ghep 3 doan 0,6651 vs ep 1 doan")
print("0,6515) cham CA BAI. Khi co hai khoi kia do, doan du loang it hon nhieu.\n")
print(f"{'cau hinh':<22}{'khoi 2 rieng':>14}{'ca bai':>12}")
print("-" * 50)
cfgs = [("dense thuan (a=1)", 1.0), (f"dinh RRF (a={best[('rrf',3)]:.1f})", best[("rrf", 3)]),
        ("rerank thuan (a=0)", 0.0)]
whole = {}
for label, a in cfgs:
    for k in (1, 2, 3):
        b2, wb = [], []
        for qid, lead, quote, concl, items in cases:
            txt = build(order_rrf(items, a), k)
            b2.append(met(quote, txt))
            full_ref = f"{lead}\n{quote}\n{concl}"
            full_hyp = f"{lead}\n{txt}\n{concl}"
            wb.append(met(full_ref, full_hyp))
        whole[(label, k)] = (statistics.fmean(b2), statistics.fmean(wb))
        print(f"{label + f' top-{k}':<22}{whole[(label,k)][0]:>14.4f}{whole[(label,k)][1]:>12.4f}")
    print()

print(f"Tong {time.time()-t0:.0f}s")
print("\n--- CHOT ALPHA TREN MAU TRAIN, KHONG CHOT BANG LB ---")
