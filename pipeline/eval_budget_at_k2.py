"""Quet lai PER_ITEM_BUDGET_SYLLABLES tai k=2 (truoc day chot o the gioi k=3).

VI SAO PHAI QUET LAI: 350 duoc chot boi E6b trong the gioi 3 doan. Sang 2 doan
thi tong ngan sach doi. Gold Khoi 2 median 187 am tiet; 2 x 350 = 700, gap gan
BON LAN gold. Vung phang 300-500 cua E6b cung do o the gioi 3 doan.

MOI HANG GHI RO: nhanh = title, alpha = 0.8, harness = CA BAI (Khoi 1+3 tu gold).

Chay: PYTHONIOENCODING=utf-8 python pipeline/eval_budget_at_k2.py
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
ALPHA = 0.8
RRF_K = 60
BUDGETS = [100, 150, 200, 250, 300, 350, 400, 500, 700]
KS = [1, 2, 3]
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
    items = expand(kl, ki, budget)
    return "\n\n".join(k.get("text", "") for k in items if k.get("text", "").strip())


cases = []
with open(L3_DIR / f"L3_rerank_train_{ARM}.jsonl", encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        lead, quote, concl = split_blocks(train[o["qid"]]["answer"])
        if not quote or len(quote.split()) < 10:
            continue
        items = [(resolve(u), u.get("dense_rank", 1 << 20), r) for r, u in enumerate(o["ranked_units"])]
        items.sort(key=lambda it: -(ALPHA / (RRF_K + it[1]) + (1 - ALPHA) / (RRF_K + it[2])))
        cases.append((lead, quote, concl, items))

gold_len = sorted(syl(c[1]) for c in cases)
print(f"n={len(cases)}  nhanh={ARM}  alpha={ALPHA}  harness=CA BAI (Khoi 1+3 tu gold)")
print(f"gold Khoi 2: median {gold_len[len(gold_len)//2]} am tiet  "
      f"p90 {gold_len[9*len(gold_len)//10]}\n", flush=True)

print(f"{'ngan sach':>10}" + "".join(f"{'k='+str(k):>12}" for k in KS) + f"{'dai k=2':>12}")
print("-" * 58)
store = {}
t0 = time.time()
for B in BUDGETS:
    row = f"{B:>10}"
    lens2 = []
    for k in KS:
        vals = []
        for lead, quote, concl, items in cases:
            parts = []
            for it in items[:k]:
                t = text_of(it[0], B)
                if t.strip():
                    parts.append(t)
            hyp = "\n\n".join(parts)
            if k == 2:
                lens2.append(syl(hyp))
            vals.append(met(f"{lead}\n{quote}\n{concl}", f"{lead}\n{hyp}\n{concl}"))
        store[(B, k)] = vals
        row += f"{statistics.fmean(vals):>12.4f}"
    lens2.sort()
    row += f"{lens2[len(lens2)//2]:>10} at"
    print(row, flush=True)

best_B = max(BUDGETS, key=lambda B: statistics.fmean(store[(B, 2)]))
print(f"\nDINH tai k=2: ngan sach {best_B} = {statistics.fmean(store[(best_B,2)]):.4f}")
d, lo, hi = paired_bootstrap(store[(350, 2)], store[(best_B, 2)])
print(f"  so voi 350 dang dat: {d:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]"
      f"{' *' if lo > 0 else ' (trong nhieu)'}")

print("\n=== PHEP SO MOT BIEN: k=3 -> k=2, cung ngan sach, cung alpha, cung nhanh ===")
for B in (350, best_B):
    d, lo, hi = paired_bootstrap(store[(B, 3)], store[(B, 2)])
    print(f"  ngan sach {B}: {statistics.fmean(store[(B,3)]):.4f} -> "
          f"{statistics.fmean(store[(B,2)]):.4f}   {d:+.4f} CI95 [{lo:+.4f}, {hi:+.4f}]"
          f"{' *' if lo > 0 else ''}")

print(f"\nTong {time.time()-t0:.0f}s")
