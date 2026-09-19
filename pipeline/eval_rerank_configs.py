"""So BON cau hinh bang METEOR cua DOAN THAT SE GIAO DI (theo thu hang).

Khac eval_t3b_hybrid_rerank.py o cho quyet dinh: script do lay doan TOT NHAT
trong toan ro -> do CHAT LUONG RO, khong phu thuoc thu hang. Ban nop that lay
TOP-N theo thu hang. Voi phat hien "reranker dim nhanh title" thi hai thuoc nay
co the le nhau, nen phai do dung thu se giao.

Bon cau hinh (2x2):
  bare  + rerank      thu tu trong file (da sap theo diem reranker)
  bare  KHONG rerank  sap lai theo dense_rank
  title + rerank
  title KHONG rerank  <- ung vien tot nhat theo T3b (64,8% hang 1)

Do o hai muc: TOP-1 va TOP-3 (V8 dang dat MAX_CONTEXTS=3).

Chay: PYTHONIOENCODING=utf-8 python pipeline/eval_rerank_configs.py
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


def text_of(row):
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
    items = expand(kl, ki, BUDGET)
    return "\n\n".join(k.get("text", "") for k in items if k.get("text", "").strip())


arms = {}
for arm in ("bare", "title"):
    d = {}
    with open(L3_DIR / f"L3_rerank_train_{arm}.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            d[o["qid"]] = o["ranked_units"]
    arms[arm] = d

CONFIGS = [
    ("bare  + rerank", "bare", False),
    ("bare  KHONG rr", "bare", True),
    ("title + rerank", "title", False),
    ("title KHONG rr", "title", True),
]

scores = {name: {"top1": [], "top3": []} for name, _, _ in CONFIGS}
lens3 = {name: [] for name, _, _ in CONFIGS}
n = 0
t0 = time.time()
for qid in arms["bare"]:
    if qid not in arms["title"]:
        continue
    quote = split_quote(train[qid]["answer"])
    if not quote:
        continue
    if len(quote.split()) < 10:
        continue
    n += 1
    for name, arm, by_dense in CONFIGS:
        units = arms[arm][qid]
        if by_dense:
            units = sorted(units, key=lambda u: u.get("dense_rank", 1 << 30))
        t1 = text_of(resolve(units[0])) if units else ""
        parts = []
        for u in units[:3]:
            tt = text_of(resolve(u))
            if tt.strip():
                parts.append(tt)
        t3 = "\n\n".join(parts)
        scores[name]["top1"].append(met(quote, t1))
        scores[name]["top3"].append(met(quote, t3))
        lens3[name].append(syl(t3))
    if n % 250 == 0:
        print(f"  {n}  {time.time()-t0:.0f}s", flush=True)

print(f"\nn={n}  (mau seed 1909, giao voi mau E4 = 0)\n")
print(f"{'cau hinh':<17}{'METEOR top-1':>14}{'METEOR top-3':>14}{'dai top-3':>12}")
print("-" * 58)
for name, _, _ in CONFIGS:
    ln = sorted(lens3[name])
    print(f"{name:<17}{statistics.fmean(scores[name]['top1']):>14.4f}"
          f"{statistics.fmean(scores[name]['top3']):>14.4f}"
          f"{ln[len(ln)//2]:>10} am tiet")

base = "bare  + rerank"
print(f"\n=== So voi moc hien tai ({base}) — bootstrap cap 2.000 lan ===")
for level in ("top1", "top3"):
    print(f"\n  {level.upper()}")
    for name, _, _ in CONFIGS:
        if name == base:
            continue
        d, lo, hi = paired_bootstrap(scores[base][level], scores[name][level])
        flag = " *" if lo > 0 else (" !" if hi < 0 else "")
        better = sum(1 for x, y in zip(scores[base][level], scores[name][level]) if y > x + 1e-9)
        worse = sum(1 for x, y in zip(scores[base][level], scores[name][level]) if y < x - 1e-9)
        print(f"    {name:<17}{d:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]{flag:<3}"
              f"  tot {better} / te {worse}")

print("\n(* = can duoi CI > 0, thang that;  ! = can tren CI < 0, thua that)")
print("\nLUU Y: top-3 o day noi don gian 3 doan, KHONG co dedup Jaccard va")
print("TARGET_TOTAL nhu build_qa_packages_v8. So tuyet doi vi the khong bang")
print("ban nop that; chenh lech giua cac cau hinh moi la thu dang doc.")
