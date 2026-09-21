"""P2 buoc 1 — Dung du lieu day cho bi-encoder muc KHOAN, bang NHAN THEO CHUOI.

Vi sao khong dung ma: `data123/` van chua duoc cap, va nhan theo ma tung dinh bug
trung `unit_id` (E1). Nhan theo chuoi mien nhiem voi bug do — no chi do do PHU CHU
giua Khoi 2 cua dap an chuan va text that cua tung Khoan.

Cach lam, moi cau:
  1. tach Khoi 2 (trich luat nguyen van) khoi dap an chuan
  2. BM25 lay top-20 van ban
  3. trong do tim Khoan phu duoc nhieu chu nhat cua Khoi 2
  4. giu lai neu do phu >= NGUONG

Hai dieu phai bao cao chu khong duoc giau:
  - ty le cau KHONG tim duoc nhan du tin -> P2 mat nen neu ty le nay cao
  - phan bo do phu -> nhan nhieu nhieu thi day ra model nhieu

THIEN LECH DA BIET: thu hep bang BM25 top-20 nen cau nao BM25 truot van ban dung
thi mat nhan. Day la thien lech ve phia "de", va no LAM HONG phan hard-negative
neu khong can than: negative lay tu cung ro BM25 la negative that, nhung positive
bi mat se khong duoc dem.

LOAI TRU: 1000 cau dang dung de danh gia (outputs/eval_1000_qids.json) bi loai
khoi tap day. Khong lam thi moi so do sau nay la do tri nho.

Chay: PYTHONIOENCODING=utf-8 python pipeline/build_khoan_pairs_string.py
"""
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import config, io_utils
from src.b2_retrieval import retrieve

TOP_K_DOCS = 20
MIN_COVERAGE = 0.45          # nguong giu nhan, tinh tren F1 (khong phai recall)
MIN_QUOTE_TOKENS = 10
OUT = config.OUTPUTS_DIR / "khoan_pairs_train.jsonl"   # nhan chon bang F1
EXCLUDE = config.OUTPUTS_DIR / "eval_1000_qids.json"

CONCLUSION_MARKERS = ("Như vậy", "Theo đó", "Do đó", "Vì vậy",
                      "Từ đó", "Như đã phân tích", "Tóm lại")
_MARKER_RE = re.compile(r"(?:^|\n)\s*("
                        + "|".join(re.escape(m) for m in CONCLUSION_MARKERS) + r")\b")


def split_blocks(answer):
    text = (answer or "").replace("\r\n", "\n").strip()
    if not text:
        return "", "", ""
    nl = text.find("\n")
    if nl == -1:
        return "", "", ""
    lead = text[:nl].strip()
    rs = nl
    while rs < len(text) and text[rs] in "\n \t":
        rs += 1
    ms = list(_MARKER_RE.finditer(text, rs))
    if not ms:
        return "", "", ""
    cut = ms[-1].start(1)
    q, c = text[rs:cut].strip(), text[cut:].strip()
    if not (lead and q and c):
        return "", "", ""
    return lead, q, c


def toks(t):
    return t.lower().split()


def coverage(gold_tokens, cand_text):
    """Recall thuan: ty le chu gold co mat trong ung vien."""
    if not gold_tokens:
        return 0.0
    cand = set(toks(cand_text))
    return sum(1 for t in gold_tokens if t in cand) / len(gold_tokens)


def f1(gold_set, cand_text):
    """F1 giua tap chu gold va tap chu ung vien.

    VI SAO KHONG DUNG RECALL DE CHON NHAN: recall DON DIEU TANG theo do dai ung
    vien - Khoan cang dai cang phu nhieu chu, nen "Khoan phu nhieu nhat" thuc ra
    la "Khoan DAI nhat". Dung la bay E6 da dong (phu chu khong co cuc tri).
    Do kiem 20/09: nhan chon bang recall cho positive dai gap 6 lan negative, va
    mot bo phan loai CHI DUNG DO DAI dat 94,3% - nhan nhu vay day mo hinh sai han.
    F1 phat ung vien dai lan man nen co cuc tri.
    """
    if not gold_set:
        return 0.0
    cand = set(toks(cand_text))
    if not cand:
        return 0.0
    inter = len(gold_set & cand)
    if not inter:
        return 0.0
    pr = inter / len(cand)
    rc = inter / len(gold_set)
    return 2 * pr * rc / (pr + rc)


print("Load...", flush=True)
t0 = time.time()
train = io_utils.load_train()
bm25 = retrieve.load_index(config.OUTPUTS_DIR / "bm25_doc_index.pkl")
parsed = io_utils.load_parsed_corpus()
excl = set(json.load(open(EXCLUDE, encoding="utf-8")))
print(f"  {time.time()-t0:.0f}s · corpus {len(parsed)} văn bản · loại trừ {len(excl)} câu đánh giá",
      flush=True)

qids = [q for q in sorted(train) if train[q].get("answer") and q not in excl]
print(f"  ứng viên để dựng nhãn: {len(qids)} câu\n", flush=True)

stats = Counter()
covs, rows = [], []
t0 = time.time()
for i, qid in enumerate(qids):
    lead, quote, concl = split_blocks(train[qid]["answer"])
    if not quote:
        stats["tách khối thất bại"] += 1
        continue
    gold = toks(quote)
    gold_set = set(gold)
    if len(gold) < MIN_QUOTE_TOKENS:
        stats["Khối 2 quá ngắn"] += 1
        continue

    best = None
    n_khoan = 0
    for d in retrieve.search(bm25, train[qid]["question"], top_k=TOP_K_DOCS):
        doc = parsed.get(str(d["context_id"]))
        if not doc:
            continue
        for dieu in doc.get("dieu", []):
            for k in dieu.get("khoan", []):
                n_khoan += 1
                sc = f1(gold_set, k.get("text", ""))
                if best is None or sc > best[0]:
                    best = (sc, k.get("khoan_id"), dieu.get("dieu_id"),
                            str(d["context_id"]), k.get("text", ""),
                            coverage(gold, k.get("text", "")))
    if best is None:
        stats["không có Khoản nào trong top-20"] += 1
        continue

    covs.append(best[0])
    if best[0] < MIN_COVERAGE:
        stats[f"F1 < {MIN_COVERAGE}"] += 1
        continue

    stats["giữ"] += 1
    rows.append({
        "qid": qid,
        "question": train[qid]["question"],
        "khoan_id": best[1],
        "dieu_id": best[2],
        "context_id": best[3],
        "f1": round(best[0], 4),
        "recall": round(best[5], 4),
        "khoan_text": best[4],
        "quote_tokens": len(gold),
    })

    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(qids)}  giữ {stats['giữ']}  {time.time()-t0:.0f}s", flush=True)

with open(OUT, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n=== KẾT QUẢ ===")
print(f"  ứng viên           {len(qids)}")
for k, v in stats.most_common():
    print(f"  {k:<34} {v:>6}  ({v/len(qids):.1%})")
print(f"\n  ĐÃ GHI {len(rows)} cặp -> {OUT}")

if covs:
    covs.sort()
    print(f"\n=== PHÂN BỐ F1 (trên {len(covs)} câu tìm được Khoản) ===")
    for p in (10, 25, 50, 75, 90):
        print(f"  phân vị {p:>2}%   {covs[int(p/100*len(covs))]:.3f}")
    print(f"  trung bình    {statistics.fmean(covs):.3f}")
    for th in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        print(f"  F1 >= {th:.2f}   {sum(1 for c in covs if c >= th):>6}"
              f"  ({sum(1 for c in covs if c >= th)/len(qids):.1%} số câu)")

if rows:
    print(f"\n=== KIỂM TÍNH ĐA DẠNG (dạy trùng lặp thì model học vẹt) ===")
    kc = Counter(r["khoan_id"] for r in rows)
    dc = Counter(r["context_id"] for r in rows)
    print(f"  Khoản khác nhau     {len(kc)} / {len(rows)} cặp")
    print(f"  văn bản khác nhau   {len(dc)}")
    print(f"  Khoản bị dùng lại nhiều nhất: {kc.most_common(3)}")
    ql = sorted(len(r["khoan_text"].split()) for r in rows)
    print(f"  độ dài Khoản (âm tiết): trung vị {ql[len(ql)//2]}  p90 {ql[9*len(ql)//10]}")
