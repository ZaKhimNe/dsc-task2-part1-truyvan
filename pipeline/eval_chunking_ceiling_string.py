"""E3a — Tran chunking do bang NHAN THEO CHUOI (khong dung ma, mien nhiem bug trung ma).

Cau hoi: Khoi 2 cua dap an chuan nam gon trong MOT Khoan, hay tran ra nhieu Khoan
cua cung mot Dieu, hay tran qua nhieu Dieu?

Khac measure_chunking_ceiling.py: script do dung nhan B0 (theo ma). O day khong
dung ma nao het — chi do do PHU CHU giua Khoi 2 va text that cua Khoan/Dieu.

Cach do: voi moi cau, BM25 lay top-20 van ban, trong do tim
  - Khoan don phu duoc nhieu chu nhat cua Khoi 2
  - Dieu phu duoc nhieu chu nhat cua Khoi 2
Chenh lech = tran cua viec mo rong sang ca Dieu.
"""
import json, random, re, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import config, io_utils
from src.b2_retrieval import retrieve

N_SAMPLE = 300
SEED = 2026
TOP_K_DOCS = 20

CONCLUSION_MARKERS = ("Như vậy", "Theo đó", "Do đó", "Vì vậy", "Từ đó", "Như đã phân tích", "Tóm lại")
_MARKER_RE = re.compile(r"(?:^|\n)\s*(" + "|".join(re.escape(m) for m in CONCLUSION_MARKERS) + r")\b")

def split_blocks(answer):
    text = (answer or "").replace("\r\n", "\n").strip()
    if not text: return "", "", ""
    nl = text.find("\n")
    if nl == -1: return "", "", ""
    lead = text[:nl].strip()
    rs = nl
    while rs < len(text) and text[rs] in "\n \t": rs += 1
    ms = list(_MARKER_RE.finditer(text, rs))
    if not ms: return "", "", ""
    cut = ms[-1].start(1)
    q, c = text[rs:cut].strip(), text[cut:].strip()
    if not (lead and q and c): return "", "", ""
    return lead, q, c

def toks(t): return t.lower().split()

def coverage(gold_tokens, cand_text):
    """Ty le chu cua gold co mat trong cand (multiset-aware, don gian)."""
    if not gold_tokens: return 0.0
    cand = set(toks(cand_text))
    return sum(1 for t in gold_tokens if t in cand) / len(gold_tokens)

print("Load train.json + BM25 index + parsed_corpus...", flush=True)
t0 = time.time()
train = io_utils.load_train()
bm25 = retrieve.load_index(config.OUTPUTS_DIR / "bm25_doc_index.pkl")
parsed = io_utils.load_parsed_corpus()
print(f"  -> {time.time()-t0:.0f}s, corpus {len(parsed)} van ban", flush=True)

qids = [q for q in sorted(train) if train[q].get("answer")]
random.Random(SEED).shuffle(qids)

rows = []
n_split_fail = 0
t0 = time.time()
for qid in qids:
    if len(rows) >= N_SAMPLE: break
    _, quote, _ = split_blocks(train[qid]["answer"])
    if not quote:
        n_split_fail += 1
        continue
    gold_tokens = toks(quote)
    if len(gold_tokens) < 10:
        continue
    docs = retrieve.search(bm25, train[qid]["question"], top_k=TOP_K_DOCS)
    best_khoan = best_dieu = 0.0
    for d in docs:
        doc = parsed.get(str(d["context_id"]))
        if not doc: continue
        for dieu in doc.get("dieu", []):
            cd = coverage(gold_tokens, dieu.get("text", ""))
            if cd > best_dieu: best_dieu = cd
            for k in dieu.get("khoan", []):
                ck = coverage(gold_tokens, k.get("text", ""))
                if ck > best_khoan: best_khoan = ck
    rows.append((qid, len(gold_tokens), best_khoan, best_dieu))
    if len(rows) % 50 == 0:
        print(f"  {len(rows)}/{N_SAMPLE}  {time.time()-t0:.0f}s", flush=True)

print(f"\nKhong tach duoc 3 khoi: {n_split_fail} cau (bo qua)")
print(f"Do duoc: n={len(rows)}\n")

kh = [r[2] for r in rows]; di = [r[3] for r in rows]
gains = [d - k for k, d in zip(kh, di)]

print("=== PHU CHU CUA KHOI 2 (1.0 = Khoi 2 nam GON trong unit do) ===")
print(f"Khoan don tot nhat : mean {statistics.fmean(kh):.3f}  median {statistics.median(kh):.3f}")
print(f"Ca Dieu tot nhat   : mean {statistics.fmean(di):.3f}  median {statistics.median(di):.3f}")
print(f"CHENH LECH         : mean {statistics.fmean(gains):+.3f}  median {statistics.median(gains):+.3f}")

for thr in (0.80, 0.90, 0.95):
    nk = sum(1 for v in kh if v >= thr); nd = sum(1 for v in di if v >= thr)
    print(f"\n  Phu >= {thr:.0%}:  Khoan don {nk:3d}/{len(rows)} = {nk/len(rows):5.1%}"
          f"   |  ca Dieu {nd:3d}/{len(rows)} = {nd/len(rows):5.1%}"
          f"   |  Dieu cuu them {nd-nk:+d} cau = {(nd-nk)/len(rows):+.1%}")

big = sum(1 for g in gains if g >= 0.10)
tiny = sum(1 for g in gains if g < 0.02)
print(f"\n=== MOT KHOAN CO DU KHONG ===")
print(f"Dieu hon Khoan >= 10 diem% (chunking Khoan LAM VO dap an): {big}/{len(rows)} = {big/len(rows):.1%}")
print(f"Dieu ~ Khoan (<2 diem%, mot Khoan DA DU)                 : {tiny}/{len(rows)} = {tiny/len(rows):.1%}")
print(f"\nTong: {time.time()-t0:.0f}s")
