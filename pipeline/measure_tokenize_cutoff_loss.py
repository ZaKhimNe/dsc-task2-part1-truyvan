"""T2 - Lo cat MAX_TOKENIZE_CHARS = 50.000 an mat bao nhieu?

VI SAO CHAY TRUOC T1: "ngoai 50" cua T1 co HAI nguyen nhan khac han nhau —
(1) xep hang khong day len duoc  -> viec cua reranker (T3)
(2) van ban CHUA BAO GIO vao chi muc vi bi cat -> viec cua MOT hang so
Neu (2) dang ke ma chua biet, con so cua T1 bi nhiem, va nguong >=50% se day
sang T4 fine-tune hang gio trong khi thu can sua la mot hang so.

CO CHE (da xac minh trong code):
  build_index() cat corpus_texts[cid][:50_000] — text THO cua VAN BAN.
  Nhanh dense nhung theo unit tu parsed_corpus.jsonl, DA chia san theo Khoan,
  nen KHONG dinh cat.
  => Thiet hai la MAT MOT TRONG HAI DUONG, khong phai mat ca hai.
     Nhung khong bang 0, vi ro hybrid dua vao BM25 cho cau nang tu khoa.

HAI VE:
  A. Phoi nhiem thuan corpus (khong can nhan): bao nhieu % Khoan cua toan kho
     nam SAU cho cat -> tran tren cua thiet hai
  B. Tren gold that: bao nhieu Khoan gold nam sau cho cat, va nhanh dense co
     voi toi duoc khong

Chay: PYTHONIOENCODING=utf-8 python pipeline/measure_tokenize_cutoff_loss.py
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
from src.b2_retrieval.retrieve import MAX_TOKENIZE_CHARS

E4_DIR = config.OUTPUTS_DIR / "e4"
N_SAMPLE = 1000
SEED = 2026

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


def cov(gt, text):
    if not gt:
        return 0.0
    s = set(text.lower().split())
    return sum(1 for x in gt if x in s) / len(gt)


CUT = MAX_TOKENIZE_CHARS
print(f"MAX_TOKENIZE_CHARS = {CUT:,}\n", flush=True)

print("Load parsed_corpus...", flush=True)
t0 = time.time()
parsed = io_utils.load_parsed_corpus()
print(f"  {len(parsed)} van ban, {time.time()-t0:.0f}s\n", flush=True)

# ---------------------------------------------------------------------------
# VE A — phoi nhiem thuan corpus, khong can nhan
# ---------------------------------------------------------------------------
print("=" * 68, flush=True)
print("VE A — phoi nhiem: bao nhieu phan kho nam SAU cho cat", flush=True)
print("=" * 68, flush=True)

n_doc = n_doc_over = 0
n_unit = n_unit_after = 0
doc_lens = []
over_keep_ratio = []
for cid, doc in parsed.items():
    n_doc += 1
    # do dai van ban = char_end lon nhat thay duoc
    maxend = 0
    units_here = []
    for dieu in doc.get("dieu", []):
        e = dieu.get("char_end") or 0
        if e > maxend:
            maxend = e
        if dieu.get("khoan"):
            for k in dieu["khoan"]:
                units_here.append(k.get("char_start"))
                ke = k.get("char_end") or 0
                if ke > maxend:
                    maxend = ke
        else:
            units_here.append(dieu.get("char_start"))
    doc_lens.append(maxend)
    if maxend > CUT:
        n_doc_over += 1
        over_keep_ratio.append(CUT / maxend)
    for cs in units_here:
        n_unit += 1
        if cs is not None and cs >= CUT:
            n_unit_after += 1

doc_lens.sort()
print(f"Van ban                     : {n_doc:,}")
print(f"  vuot {CUT:,} ky tu        : {n_doc_over:,} = {n_doc_over/n_doc:.2%}")
print(f"  do dai: median {doc_lens[n_doc//2]:,}  p90 {doc_lens[9*n_doc//10]:,}  "
      f"max {doc_lens[-1]:,}")
if over_keep_ratio:
    over_keep_ratio.sort()
    print(f"  van ban bi cat GIU LAI  : median {statistics.median(over_keep_ratio):.1%}  "
          f"min {over_keep_ratio[0]:.2%}  (tuc mat phan con lai)")
print(f"\nUnit (Khoan/Dieu)           : {n_unit:,}")
print(f"  nam SAU cho cat           : {n_unit_after:,} = {n_unit_after/n_unit:.2%}  "
      f"<-- BM25 khong bao gio thay")
print("  (nhanh dense VAN thay, vi nhung theo unit tu parsed_corpus)\n", flush=True)

# ---------------------------------------------------------------------------
# VE B — tren gold that
# ---------------------------------------------------------------------------
print("=" * 68, flush=True)
print("VE B — gold that: bao nhieu Khoan gold nam sau cho cat", flush=True)
print("=" * 68, flush=True)

# dinh vi gold bang RO DENSE cua E4 (dense KHONG dinh cat -> khong vong tron
# voi cai dang do). Gold khong nam trong ro dense thi khong dinh vi duoc,
# bao rieng thanh mot nhom.
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

basket = {}
with open(E4_DIR / "top50_title_qa_train.jsonl", encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        basket[o["qid"]] = o["rows"]

train = io_utils.load_train()
qids = [q for q in sorted(basket) if train.get(q, {}).get("answer")]
random.Random(SEED).shuffle(qids)

n_located = n_after = n_not_located = 0
after_examples = []
t0 = time.time()
done = 0
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
    best = (0.0, None)
    for r in basket[qid]:
        cid, di, ki = row_map[r]
        doc = parsed.get(cid)
        if not doc or di >= len(doc.get("dieu", [])):
            continue
        dieu = doc["dieu"][di]
        kl = dieu.get("khoan", [])
        unit = kl[ki] if (ki >= 0 and ki < len(kl)) else dieu
        c = cov(gt, unit.get("text", ""))
        if c > best[0]:
            best = (c, (cid, unit))
    if best[0] < 0.70 or best[1] is None:
        n_not_located += 1
        continue
    n_located += 1
    cid, unit = best[1]
    cs = unit.get("char_start")
    if cs is not None and cs >= CUT:
        n_after += 1
        if len(after_examples) < 5:
            after_examples.append((qid, cid, cs))
    if done % 200 == 0:
        print(f"  {done}/{N_SAMPLE}  {time.time()-t0:.0f}s", flush=True)

print(f"\nDo tren {done} cau (nhan theo chuoi, nguong phu 70%):")
print(f"  dinh vi duoc gold trong ro dense : {n_located} = {n_located/done:.1%}")
print(f"  KHONG dinh vi duoc               : {n_not_located} = {n_not_located/done:.1%}"
      f"  (gold ngoai top-50 dense — thuoc bai toan cua T1)")
if n_located:
    print(f"\n  Trong so dinh vi duoc:")
    print(f"    gold nam SAU cho cat  : {n_after} = {n_after/n_located:.2%}"
          f"   <-- BM25 mu, chi dense cuu")
    print(f"    gold nam TRUOC cho cat: {n_located-n_after} = {(n_located-n_after)/n_located:.2%}")
for qid, cid, cs in after_examples:
    print(f"      vd qid={qid} context={cid} char_start={cs:,}")

print("\n--- DOC SO ---")
print("Ve A la TRAN TREN (phoi nhiem toan kho). Ve B la uoc luong tren gold that.")
print("Ca hai deu chi do nhom gold MA DENSE DA VOI TOI. Gold khong ai voi toi")
print("duoc thi nam o nhom 'khong dinh vi duoc' — do la bai toan cua T1, khong")
print("phai cua cho cat.")
