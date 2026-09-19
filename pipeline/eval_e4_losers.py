"""T5 - Soi nhom cau ma E4 (nhung kem tieu de) lam TE DI.

E4 rong duong nhung 75/1.000 cau te di — khac han E5 (8 thang/1 thua, gan mot chieu).

CO CHE DOAN DUOC: gan tieu de lam vector mo ta CHU DE DIEU ro hon, noi dung rieng
cua KHOAN mo di. Nen cau can mot Khoan LECH khoi chu de chung se te di.

KIEM HAI DIEU:
  1. Nhom thua co roi vao Dieu tieu de DAI hon khong?
  2. Khoan gold cua nhom thua co LECH khoi tieu de Dieu nhieu hon khong?
     (do bang do trung chu giua text Khoan gold va tieu de Dieu)

NEU KHOP: cach va KHONG phai chon mot ben. Cham CA HAI vector roi lay diem cao
hon — giu +4,40 ma khong mat nhom thua. Re, vi ca hai ma tran da nam san trong
output kernel E4.

Chay: PYTHONIOENCODING=utf-8 python pipeline/eval_e4_losers.py
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
print(f"  {time.time()-t0:.0f}s\n", flush=True)


def deliverable(row, gt):
    cid, di, ki = row_map[row]
    doc = parsed.get(cid)
    if not doc or di >= len(doc.get("dieu", [])):
        return 0.0, "", None, None
    dieu = doc["dieu"][di]
    kl = dieu.get("khoan", [])
    if ki < 0 or not kl:
        t = dieu.get("text", "")
        return cov(gt, t), t, dieu, dieu
    items = expand(kl, ki, BUDGET)
    t = "\n\n".join(k.get("text", "") for k in items if k.get("text", "").strip())
    return cov(gt, t), t, dieu, kl[ki]


def best_over(rows, gt):
    best = (0.0, "", None, None)
    for r in rows:
        c, t, dieu, unit = deliverable(r, gt)
        if c > best[0]:
            best = (c, t, dieu, unit)
    return best


arms = {}
for arm in ("bare", "title"):
    d = {}
    with open(E4_DIR / f"top50_{arm}_qa_train.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            d[o["qid"]] = o["rows"]
    arms[arm] = d

qids = [q for q in sorted(arms["bare"]) if q in arms["title"] and train.get(q, {}).get("answer")]
random.Random(SEED).shuffle(qids)

groups = {"thua": [], "thang": [], "khong doi": []}
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
    cb, tb, _, _ = best_over(arms["bare"][qid], gt)
    ct, tt, dieu_t, unit_t = best_over(arms["title"][qid], gt)
    mb, mt = met(quote, tb), met(quote, tt)
    # dac trung cua nhom: lay tu DIEU ma nhanh title chon
    if dieu_t is None:
        continue
    td = (dieu_t.get("dieu_tieu_de") or "").strip()
    gold_text = unit_t.get("text", "") if unit_t else ""
    # do lech giua Khoan va tieu de Dieu: ty le chu cua tieu de xuat hien trong Khoan
    td_toks = set(td.lower().split())
    k_toks = set(gold_text.lower().split())
    overlap = len(td_toks & k_toks) / len(td_toks) if td_toks else None
    rec = {"qid": qid, "title_syl": syl(td), "overlap": overlap, "delta": mt - mb}
    if mt < mb - 1e-9:
        groups["thua"].append(rec)
    elif mt > mb + 1e-9:
        groups["thang"].append(rec)
    else:
        groups["khong doi"].append(rec)
    if done % 250 == 0:
        print(f"  {done}/{N_SAMPLE}  {time.time()-t0:.0f}s", flush=True)

print(f"\nn={done}")
print(f"{'nhom':<12}{'so cau':>8}{'tieu de (am tiet)':>22}{'trung chu tieu de<->Khoan':>28}")
print("-" * 72)
for name in ("thang", "thua", "khong doi"):
    g = groups[name]
    if not g:
        continue
    ts = sorted(r["title_syl"] for r in g)
    ov = sorted(r["overlap"] for r in g if r["overlap"] is not None)
    ov_s = (f"median {statistics.median(ov):.3f}  mean {statistics.fmean(ov):.3f}"
            if ov else "—")
    print(f"{name:<12}{len(g):>8}   median {ts[len(ts)//2]:>3}  mean {statistics.fmean(ts):>5.1f}"
          f"   {ov_s:>28}")

# ty le tieu de DAI trong tung nhom
print(f"\n{'nhom':<12}{'tieu de >20 am tiet':>22}{'>30 am tiet':>16}")
print("-" * 52)
for name in ("thang", "thua", "khong doi"):
    g = groups[name]
    if not g:
        continue
    n20 = sum(1 for r in g if r["title_syl"] > 20)
    n30 = sum(1 for r in g if r["title_syl"] > 30)
    print(f"{name:<12}{n20:>8} = {n20/len(g):>6.1%}{n30:>10} = {n30/len(g):>6.1%}")

print("\n=== 8 CAU THUA NANG NHAT ===")
for r in sorted(groups["thua"], key=lambda x: x["delta"])[:8]:
    ov = f"{r['overlap']:.3f}" if r["overlap"] is not None else "—"
    print(f"  qid={r['qid']:<9} METEOR {r['delta']:+.4f}  tieu de {r['title_syl']:>3} am tiet"
          f"  trung chu {ov}")

print("\n--- DOC SO ---")
print("Neu nhom THUA co tieu de DAI hon va TRUNG CHU THAP hon nhom THANG,")
print("thi co che doan dung: tieu de lam mo noi dung rieng cua Khoan.")
print("Khi do cach va la cham CA HAI vector roi lay diem cao hon, khong phai chon mot ben.")
