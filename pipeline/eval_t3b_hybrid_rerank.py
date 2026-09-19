"""T3b - Cham bare vs title tren RO HYBRID DA RERANK, mau doc lap voi E4.

Mau: 1.000 cau qa_train, seed 1909, GIAO VOI MAU E4 = 0. Dung lai mau E4 la do
tren tap da dung de CHON title thang bare -> ket qua dep hon thuc te.

Do ba thu tren cung mot ro:
  1. Hang gold TRUOC rerank (theo dense_rank) va SAU rerank
  2. METEOR cua thu SE GIAO DI (mo rong ngan sach 400)
  3. Chenh lech title - bare, bootstrap cap 2.000 lan

NGUONG AM TINH GIA DAT TRUOC (ghi truoc khi doc so):
  Ro hybrid dung union BM25 ∥ dense, ma BM25 dang mu 25,39% unit (T2).
  -> Neu title KHONG thang: KHONG dong huong. Phai do lai sau khi va BM25.
  -> Neu title thang: ket luan dung, vi thang tren ro kem thi ro tot cang thang.

Chay: PYTHONIOENCODING=utf-8 python pipeline/eval_t3b_hybrid_rerank.py
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
COV_THRESHOLD = 0.70
N_BOOTSTRAP = 2000
SEED = 42

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

# 60-64% ung vien trong ro hybrid co row=-1: chung den tu nhanh BM25
# (expand_to_units khong gan row), va union_units giu ban GAP DAU TIEN nen ban
# tu vung thang suat. Phai lui ve tra bang unit_id cho nhung ca do — neu khong,
# phep cham BO QUA 60% ro va bao "gold ngoai 50" mot cach gia tao.
uid_map = {}
for _pos, (_cid, _di, _ki) in enumerate(row_map):
    _doc = parsed.get(_cid)
    if not _doc or _di >= len(_doc.get("dieu", [])):
        continue
    _dieu = _doc["dieu"][_di]
    _kl = _dieu.get("khoan", [])
    _uid = _kl[_ki]["khoan_id"] if (0 <= _ki < len(_kl)) else _dieu.get("dieu_id")
    uid_map.setdefault(_uid, _pos)
print(f"  {time.time()-t0:.0f}s  | bang tra unit_id: {len(uid_map)} ma\n", flush=True)


def unit_of(row):
    if row is None or row < 0 or row >= len(row_map):
        return None, None
    cid, di, ki = row_map[row]
    doc = parsed.get(cid)
    if not doc or di >= len(doc.get("dieu", [])):
        return None, None
    dieu = doc["dieu"][di]
    kl = dieu.get("khoan", [])
    return dieu, (kl[ki] if (0 <= ki < len(kl)) else dieu)


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


def gold_rank(units, gt):
    """Hang (1-based) cua ung vien dau tien phu >= nguong; None neu khong co."""
    for pos, u in enumerate(units):
        row = u.get("row", -1)
        if row is None or row < 0:
            row = uid_map.get(u.get("unit_id"), -1)
        dieu, unit = unit_of(row)
        if unit is None:
            continue
        if cov(gt, unit.get("text", "")) >= COV_THRESHOLD or \
           cov(gt, dieu.get("text", "")) >= COV_THRESHOLD:
            return pos + 1
    return None


arms = {}
for arm in ("bare", "title"):
    d = {}
    with open(L3_DIR / f"L3_rerank_train_{arm}.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            d[o["qid"]] = o["ranked_units"]
    arms[arm] = d
    print(f"{arm}: {len(d)} cau", flush=True)

qids = [q for q in arms["bare"] if q in arms["title"]]
res = {a: {"rank_after": [], "rank_before": [], "meteor": []} for a in ("bare", "title")}
n = 0
t0 = time.time()
for qid in qids:
    quote = split_quote(train[qid]["answer"])
    if not quote:
        continue
    gt = quote.lower().split()
    if len(gt) < 10:
        continue
    n += 1
    for arm in ("bare", "title"):
        units = arms[arm][qid]
        res[arm]["rank_after"].append(gold_rank(units, gt))
        before = sorted(units, key=lambda u: u.get("dense_rank", 1 << 30))
        res[arm]["rank_before"].append(gold_rank(before, gt))
        bc, bt = 0.0, ""
        for u in units:
            _r = u.get("row", -1)
            if _r is None or _r < 0:
                _r = uid_map.get(u.get("unit_id"), -1)
            if _r < 0:
                continue
            c, t = deliverable(_r, gt)
            if c > bc:
                bc, bt = c, t
        res[arm]["meteor"].append(met(quote, bt))
    if n % 250 == 0:
        print(f"  {n}  {time.time()-t0:.0f}s", flush=True)

BUCKETS = [("hang 1", 1, 1), ("hang 2-3", 2, 3), ("hang 4-10", 4, 10), ("hang 11-50", 11, 50)]
print(f"\nn={n}  (mau seed 1909, giao voi mau E4 = 0)\n")

for arm in ("bare", "title"):
    for phase in ("rank_before", "rank_after"):
        ranks = res[arm][phase]
        inside = [r for r in ranks if r is not None]
        label = "TRUOC rerank (thu tu dense)" if phase == "rank_before" else "SAU rerank"
        print(f"=== {arm.upper()} — {label} ===")
        for name, lo, hi in BUCKETS:
            c = sum(1 for r in inside if lo <= r <= hi)
            print(f"  {name:<12}{c:>5} = {c/len(ranks):>6.1%}")
        print(f"  {'NGOAI 50':<12}{len(ranks)-len(inside):>5} = "
              f"{(len(ranks)-len(inside))/len(ranks):>6.1%}")
        if inside:
            print(f"  hang mean {statistics.fmean(inside):.2f}")
        print()

print("=== HANG 1: TRUOC vs SAU rerank ===")
for arm in ("bare", "title"):
    b = sum(1 for r in res[arm]["rank_before"] if r == 1) / n
    a = sum(1 for r in res[arm]["rank_after"] if r == 1) / n
    print(f"  {arm:<6} {b:>6.1%} -> {a:>6.1%}   rerank {a-b:+.1%}")

print("\n=== TITLE vs BARE ===")
for arm in ("bare", "title"):
    print(f"  {arm:<6} METEOR {statistics.fmean(res[arm]['meteor']):.4f}")
d, lo, hi = paired_bootstrap(res["bare"]["meteor"], res["title"]["meteor"])
flag = " *" if lo > 0 else (" !" if hi < 0 else "")
print(f"  chenh {d:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]{flag}")
better = sum(1 for x, y in zip(res["bare"]["meteor"], res["title"]["meteor"]) if y > x + 1e-9)
worse = sum(1 for x, y in zip(res["bare"]["meteor"], res["title"]["meteor"]) if y < x - 1e-9)
print(f"  tot hon {better}  te hon {worse}  khong doi {n-better-worse}")

a1 = [1.0 if r == 1 else 0.0 for r in res["bare"]["rank_after"]]
b1 = [1.0 if r == 1 else 0.0 for r in res["title"]["rank_after"]]
d, lo, hi = paired_bootstrap(a1, b1)
print(f"  hang 1 sau rerank: {statistics.fmean(a1):.1%} -> {statistics.fmean(b1):.1%}  "
      f"chenh {d*100:+.2f}d  CI95 [{lo*100:+.2f}, {hi*100:+.2f}]")

print("\n--- NGUONG AM TINH GIA (dat truoc khi doc so) ---")
print("Ro hybrid dung BM25 dang mu 25,39% unit (T2). Neu title KHONG thang thi")
print("KHONG dong huong — phai do lai sau khi va BM25. Neu thang thi ket luan dung.")
