"""T11 — Tap danh gia bi LOC SAN dat gia bao nhieu?

Phat hien 6.7: 1.000 cau dang dung de danh gia tach duoc ba khoi 100%, trong khi
dan so chung chi 53,4%. Nen moi so offline cua Task 2 do tren mot nua "de".
Private khong co bo loc do.

Cau hoi: hai nhom do co thuc su khac diem khong, va khac bao nhieu?

NGUYEN LIEU DA CO SAN, KHONG CAN THEM GPU RERANK:
kernel E4 da nhung CA 7.000 cau train -> outputs/e4/top50_title_qa_train.jsonl
co du 7.000 dong. Do la ro dense top-50 cho MOI cau, khong chi 1.000 cau da loc.

Dung alpha=1.0: cong thuc tron la
    s = alpha/(60+dense_rank) + (1-alpha)/(60+rerank_rank)
alpha=1 lam so hang thu hai TRIET TIEU, nen khong can diem reranker.
alpha=1 kem alpha=0,7 khoang 0,0096 tren harness — nhung ta do CHENH LECH GIUA
HAI NHOM, nen chi can hai nhom cung cau hinh, khong can cau hinh tot nhat.

KHONG dung build_qa_packages_public_v8_tieu_de_khoan.py: file do thuoc phien kia
theo luat chia file. Script nay tu dung goi, dung dung logic expand() cua cac
script eval trong pipeline/ (cung phien).

Chay: PYTHONIOENCODING=utf-8 python pipeline/build_packages_loc_vs_khonglọc.py
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import config, io_utils

SEED = 2026
N_PER_GROUP = 200
BUDGET = 200
K = 2
E4 = config.OUTPUTS_DIR / "e4" / "top50_title_qa_train.jsonl"
OUT = config.OUTPUTS_DIR / "qa_packages_loc_vs_khongloc_a1_k2_b200.json"

MARKERS = ("Như vậy", "Theo đó", "Do đó", "Vì vậy", "Từ đó", "Như đã phân tích", "Tóm lại")
_RE = re.compile(r"(?:^|\n)\s*(" + "|".join(re.escape(m) for m in MARKERS) + r")\b")


def splittable(answer):
    t = (answer or "").replace("\r\n", "\n").strip()
    if not t or "\n" not in t:
        return False
    nl = t.find("\n")
    rs = nl
    while rs < len(t) and t[rs] in "\n \t":
        rs += 1
    ms = list(_RE.finditer(t, rs))
    if not ms:
        return False
    cut = ms[-1].start(1)
    return bool(t[:nl].strip() and t[rs:cut].strip() and t[cut:].strip())


def syl(t):
    return len(t.split())


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
parsed = io_utils.load_parsed_corpus()
train = io_utils.load_train()
ev = set(json.load(open(config.OUTPUTS_DIR / "eval_1000_qids.json", encoding="utf-8")))

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
print(f"  corpus {len(row_map)} unit", flush=True)

baskets = {}
with open(E4, encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        baskets[str(o["qid"])] = o["rows"]
print(f"  ro E4: {len(baskets)} cau", flush=True)


def context_of(row):
    """Dung mot context theo dung schema generator doc (pipeline.py:79-99)."""
    if row < 0 or row >= len(row_map):
        return None
    cid, di, ki = row_map[row]
    doc = parsed.get(cid)
    if not doc or di >= len(doc.get("dieu", [])):
        return None
    dieu = doc["dieu"][di]
    kl = dieu.get("khoan", [])
    title = (dieu.get("dieu_tieu_de") or "").strip()
    if ki < 0 or not kl:
        text = dieu.get("text", "")
        clause = None
    else:
        items = expand(kl, ki, BUDGET)
        text = "\n\n".join(k.get("text", "") for k in items if k.get("text", "").strip())
        clause = kl[ki].get("khoan_so")
    if not text.strip():
        return None
    return {
        "text": text,
        "article": dieu.get("dieu_so"),
        "article_title": title or None,
        "clause": clause,
        "document": doc.get("name"),
        "document_number": None,
        "retrieval_score": None,
    }


# --- chon hai nhom ---------------------------------------------------------
pool_loc, pool_khong = [], []
for qid, r in train.items():
    if qid not in baskets or not r.get("answer"):
        continue
    if qid in ev:
        pool_loc.append(qid)                      # nam trong tap danh gia (100% tach duoc)
    elif not splittable(r["answer"]):
        pool_khong.append(qid)                    # KHONG tach duoc — nhom chua bao gio do
rng = random.Random(SEED)
rng.shuffle(pool_loc)
rng.shuffle(pool_khong)
sel_loc = pool_loc[:N_PER_GROUP]
sel_khong = pool_khong[:N_PER_GROUP]
print(f"\nnhom LOC (trong tap danh gia)  : {len(pool_loc)} ung vien -> lay {len(sel_loc)}")
print(f"nhom KHONG LOC (khong tach duoc): {len(pool_khong)} ung vien -> lay {len(sel_khong)}",
      flush=True)

rows, skipped = [], 0
groups = {}
for group, sel in (("loc", sel_loc), ("khongloc", sel_khong)):
    for qid in sel:
        ctxs = []
        for row in baskets[qid]:          # rows da theo thu tu dense -> alpha=1 la chinh no
            c = context_of(row)
            if c:
                ctxs.append(c)
            if len(ctxs) >= K:
                break
        if not ctxs:
            skipped += 1
            continue
        groups[str(qid)] = group
        rows.append({
            "id": str(qid),
            "question": train[qid]["question"],
            "reference_answer": train[qid]["answer"],
            "contexts": ctxs,
        })

json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
json.dump(groups, open(config.OUTPUTS_DIR / "loc_vs_khongloc_groups.json", "w",
                       encoding="utf-8"), ensure_ascii=False, indent=0)

import collections
gc = collections.Counter(groups.values())
print(f"\nĐÃ GHI {len(rows)} câu -> {OUT.name}   (bỏ {skipped} câu không dựng được context)")
print(f"  nhóm loc {gc['loc']}   nhóm khongloc {gc['khongloc']}")
nl = sorted(syl(r["reference_answer"]) for r in rows if groups[r["id"]] == "loc")
nk = sorted(syl(r["reference_answer"]) for r in rows if groups[r["id"]] == "khongloc")
print(f"\n  độ dài đáp án chuẩn (âm tiết) — trung vị: loc {nl[len(nl)//2]}  khongloc {nk[len(nk)//2]}")
cl = sorted(sum(syl(c["text"]) for c in r["contexts"]) for r in rows if groups[r["id"]] == "loc")
ck = sorted(sum(syl(c["text"]) for c in r["contexts"]) for r in rows if groups[r["id"]] == "khongloc")
print(f"  độ dài context (âm tiết)      — trung vị: loc {cl[len(cl)//2]}  khongloc {ck[len(ck)//2]}")
print("\n  Hai nhóm đi qua ĐÚNG một bộ dựng, đúng một cấu hình (alpha=1, k=2, 200).")
print("  Chênh lệch điểm giữa chúng do đó quy được về khác biệt của CHÍNH CÂU HỎI.")
