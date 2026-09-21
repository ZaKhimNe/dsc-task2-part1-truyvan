"""P1 — CHAM LAI CHI TOP-W: reranker chi duoc noi trong cua so W dau, ngoai do
giu nguyen thu tu nen (dense).

Vi sao cau hinh nay chua ai thu, va vi sao no dang thu:
  - T3 do: reranker ghi de ca 50 -> tren nhanh title no LAM HAI (64,8 -> 60,7)
  - T6 do: 71% du dia nam trong top-10 (tran top-10 +0,0822 / tran top-50 +0,1159)
  -> hai so cung chi ve: cho reranker noi o phan DAU, cam no dung toi phan duoi.

W=50 la hanh vi hien tai (reranker noi tren toan ro). W nho hon = han che dan.

NGUONG DAT TRUOC (ca ba phai dat, khong thi dong huong):
  1. cau hinh tot nhat hon san xuat (W=50, alpha=0,7) >= +0,010
  2. can duoi CI95 > 0
  3. thang tren nua GIU RIENG  <- bat buoc, vi quet 24 o thi thien lech chon lon

Bao cao loi ich bang KIEM CHEO HAI NUA, khong phai bootstrap toan mau.
Bai hoc alpha: quet toan mau cho +0,0082, kiem cheo chi con +0,006.

Chay: PYTHONIOENCODING=utf-8 python pipeline/eval_cua_so_rerank.py
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
RRF_K = 60
BUDGET = 200
K = 2
WINDOWS = [5, 10, 20, 50]
ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BASE = (50, 0.7)          # cau hinh san xuat V9
N_BOOTSTRAP = 2000
SEED = 42
SPLIT_SEED = 1909

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
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(N_BOOTSTRAP))
    return statistics.fmean(diffs), means[int(0.025 * N_BOOTSTRAP)], means[int(0.975 * N_BOOTSTRAP)]


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
        units = [(resolve(u), u.get("dense_rank", 1 << 20), r)
                 for r, u in enumerate(o["ranked_units"])]
        cases.append((lead, quote, concl, units))

print(f"n={len(cases)}  nhanh={ARM}  ngan sach={BUDGET}  k={K}")
print(f"harness=CA BAI (Khoi 1+3 tu gold)\n", flush=True)


def order(units, W, alpha):
    """Tron alpha CHI trong W ung vien dau theo thu tu NEN (dense_rank).
    Ngoai W: giu nguyen thu tu nen, khong cho reranker dung toi."""
    base = sorted(units, key=lambda it: it[1])          # thu tu nen = dense_rank
    head, tail = base[:W], base[W:]
    head = sorted(head, key=lambda it: -(alpha / (RRF_K + it[1])
                                         + (1 - alpha) / (RRF_K + it[2])))
    return head + tail


# cache text theo row de khong dung lai text_of
tcache = {}


def body(units, W, alpha):
    parts = []
    for it in order(units, W, alpha)[:K]:
        t = tcache.get(it[0])
        if t is None:
            t = tcache[it[0]] = text_of(it[0], BUDGET)
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts)


scores = {}
t0 = time.time()
for W in WINDOWS:
    for A in ALPHAS:
        if W == 50 and A == 1.0:
            pass  # van tinh, lam moc "khong reranker"
        vals = []
        for lead, quote, concl, units in cases:
            vals.append(met(f"{lead}\n{quote}\n{concl}",
                            f"{lead}\n{body(units, W, A)}\n{concl}"))
        scores[(W, A)] = vals
    print(f"  W={W} xong  {time.time()-t0:.0f}s", flush=True)

print(f"\n{'W \\ alpha':>10}" + "".join(f"{a:>10.1f}" for a in ALPHAS))
print("-" * (10 + 10 * len(ALPHAS)))
for W in WINDOWS:
    row = f"{W:>10}"
    for A in ALPHAS:
        m = statistics.fmean(scores[(W, A)])
        row += f"{m:>10.4f}"
    print(row + ("   <- hien tai" if W == 50 else ""))

base_vals = scores[BASE]
print(f"\nsan xuat (W={BASE[0]}, alpha={BASE[1]}) = {statistics.fmean(base_vals):.4f}")

best = max(scores, key=lambda k: statistics.fmean(scores[k]))
d, lo, hi = paired_bootstrap(base_vals, scores[best])
print(f"tot nhat toan mau: W={best[0]} alpha={best[1]}  {statistics.fmean(scores[best]):.4f}"
      f"   {d:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]")

print("\n=== KIEM CHEO HAI NUA (chon tren nua A, cham tren nua B va nguoc lai) ===")
idx = list(range(len(cases)))
random.Random(SPLIT_SEED).shuffle(idx)
half = len(idx) // 2
folds = [(idx[:half], idx[half:]), (idx[half:], idx[:half])]
gains = []
for fi, (fit, test) in enumerate(folds, 1):
    pick = max(scores, key=lambda k: statistics.fmean(scores[k][i] for i in fit))
    g = statistics.fmean(scores[pick][i] - base_vals[i] for i in test)
    gains.append(g)
    print(f"  nua {fi}: chon W={pick[0]} alpha={pick[1]}"
          f"  ->  loi tren nua giu rieng {g:+.4f}")
cv = statistics.fmean(gains)
print(f"\n  LOI THEO KIEM CHEO = {cv:+.4f}   (bootstrap toan mau = {d:+.4f})")
print(f"  thien lech do chon  = {d - cv:+.4f}")

print("\n=== NGUONG DAT TRUOC ===")
c1, c2, c3 = d >= 0.010, lo > 0, cv > 0
print(f"  1. loi >= +0,010            {'DAT' if c1 else 'KHONG'}  ({d:+.4f})")
print(f"  2. can duoi CI95 > 0        {'DAT' if c2 else 'KHONG'}  ({lo:+.4f})")
print(f"  3. thang tren nua giu rieng {'DAT' if c3 else 'KHONG'}  ({cv:+.4f})")
print(f"\n  => {'DI TIEP' if (c1 and c2 and c3) else 'DONG HUONG'}")
