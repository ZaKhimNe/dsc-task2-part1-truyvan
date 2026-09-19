"""E2 - Bat `union_units(dedup_by_row=True)` thi duoc them gi?

Do TRUC TIEP tren ro dense that (outputs/e4/), mo phong dung phep loai trung
cua union_units:

  KHONG VA (hom nay): giu ban GAP DAU TIEN cho moi unit_id -> cac khoan con
                      anh em mang cung ma bi vut, du dense da cham rieng
  CO VA            : giu ca 50 hang, vi khoa la `row` chu khong phai `unit_id`

Cung mot ro dau vao, cung bo chon, cung thuoc do -> chenh lech la phan
`dedup_by_row=True` thu ve, khong lan voi bien nao khac.

Chay: PYTHONIOENCODING=utf-8 python pipeline/eval_dedup_by_row_gain.py
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
row_uid = []
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
print(f"  {len(row_map)} hang, {time.time()-t0:.0f}s", flush=True)


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


def best_over(rows, gt):
    bc, bt = 0.0, ""
    for r in rows:
        c, t = deliverable(r, gt)
        if c > bc:
            bc, bt = c, t
    return bc, bt


for arm in ("bare", "title"):
    path = E4_DIR / f"top50_{arm}_qa_train.jsonl"
    basket = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            basket[o["qid"]] = o["rows"]

    qids = [q for q in sorted(basket) if train.get(q, {}).get("answer")]
    random.Random(SEED).shuffle(qids)

    c_no, c_fix, m_no, m_fix, n_lost = [], [], [], [], []
    t0 = time.time()
    for qid in qids:
        if len(c_no) >= N_SAMPLE:
            break
        quote = split_quote(train[qid]["answer"])
        if not quote:
            continue
        gt = quote.lower().split()
        if len(gt) < 10:
            continue
        rows = basket[qid]
        # mo phong union_units hom nay: giu ban gap DAU TIEN cho moi unit_id
        seen = set()
        kept = []
        for r in rows:
            uid = row_uid[r]
            if uid in seen:
                continue
            seen.add(uid)
            kept.append(r)
        n_lost.append(len(rows) - len(kept))
        a, ta = best_over(kept, gt)
        b, tb = best_over(rows, gt)
        c_no.append(a)
        c_fix.append(b)
        m_no.append(met(quote, ta))
        m_fix.append(met(quote, tb))

    n = len(c_no)
    print(f"\n=== NHANH {arm.upper()} — n={n} ===")
    print(f"  Suat mat vi trung ma: mean {statistics.fmean(n_lost):.2f}/50  "
          f"so cau mat >=1: {sum(1 for x in n_lost if x>=1)}/{n}")
    d, lo, hi = paired_bootstrap(c_no, c_fix)
    print(f"  Phu chu : {statistics.fmean(c_no):.4f} -> {statistics.fmean(c_fix):.4f}  "
          f"chenh {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    d, lo, hi = paired_bootstrap(m_no, m_fix)
    flag = " *" if lo > 0 else ""
    print(f"  METEOR  : {statistics.fmean(m_no):.4f} -> {statistics.fmean(m_fix):.4f}  "
          f"chenh {d:+.4f} [{lo:+.4f}, {hi:+.4f}]{flag}")
    better = sum(1 for x, y in zip(m_no, m_fix) if y > x + 1e-9)
    worse = sum(1 for x, y in zip(m_no, m_fix) if y < x - 1e-9)
    print(f"  tot hon: {better}  te hon: {worse}  khong doi: {n-better-worse}")

print("\n(* = can duoi CI > 0 -> va co loi that)")
print("Luu y: day chi la nhanh DENSE. Ro san xuat co them nhanh BM25 nen phan")
print("mat suat co the khac; nhung chieu va do lon cua hieu ung thi doc duoc.")
