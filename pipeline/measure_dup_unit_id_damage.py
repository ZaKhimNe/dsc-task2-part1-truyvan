"""E1 — do thiet hai that cua bug ma Khoan trung (gold-free).

Duyet parsed_corpus.jsonl THEO DUNG thu tu cua kaggle_layer2/embed_corpus_notebook.py
CELL 4, de danh so hang khop voi hang cua ma tran embedding.

Tra loi 3 cau, khong can nhan gold:
  1. Ban sao nam o dau: cung Dieu / khac Dieu cung van ban / khac van ban?
     -> cung Dieu = thiet hai nho (text lien quan, V8 gop lien ke cung lay)
     -> khac van ban = thiet hai lon (giao cho C doan cua luat khac han)
  2. Text lech bao nhieu giua hang CHAM DIEM (cuoi) va hang GIAO TEXT (dau)?
  3. Ma nao te nhat (nhieu ban sao nhat).
"""
import json, statistics, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import config
CORPUS = config.DATA_DIR / "parsed_corpus.jsonl"

def units_in_record(r):
    """DUNG thu tu CELL 4 cua embed_corpus_notebook.py."""
    out = []
    if not r.get("dieu"):
        return out
    if r.get("parse_status") == "fallback":
        d0 = r["dieu"][0]
        out.append((d0.get("dieu_id") or r["context_id"], d0.get("text", ""), None, r["context_id"]))
        return out
    for dieu in r["dieu"]:
        if dieu.get("khoan"):
            for k in dieu["khoan"]:
                out.append((k["khoan_id"], k.get("text", ""), dieu.get("dieu_id"), r["context_id"]))
        else:
            out.append((dieu["dieu_id"], dieu.get("text", ""), dieu.get("dieu_id"), r["context_id"]))
    return out

# --- Pass 1: dem ma ---
t0 = time.time()
counts = defaultdict(int)
n_rows = 0
with open(CORPUS, encoding="utf-8") as f:
    for ln, line in enumerate(f, 1):
        if not line.strip():
            continue
        for uid, _txt, _did, _cid in units_in_record(json.loads(line)):
            counts[uid] += 1
            n_rows += 1
        if ln % 2000 == 0:
            print(f"  pass1 {ln} van ban, {n_rows} hang, {time.time()-t0:.0f}s", flush=True)

dup_ids = {u for u, c in counts.items() if c > 1}
shadowed = n_rows - len(counts)
print(f"\n=== TONG QUAN ===")
print(f"Tong hang (= hang ma tran embedding) : {n_rows}")
print(f"Ma duy nhat                          : {len(counts)}")
print(f"Hang bi che (khong tra duoc bang ma) : {shadowed} = {shadowed/n_rows:.1%}")
print(f"Ma bi trung                          : {len(dup_ids)}")

# --- Pass 2: gom text cua ma trung ---
occ = defaultdict(list)   # uid -> [(dieu_id, context_id, text)]
with open(CORPUS, encoding="utf-8") as f:
    for ln, line in enumerate(f, 1):
        if not line.strip():
            continue
        for uid, txt, did, cid in units_in_record(json.loads(line)):
            if uid in dup_ids:
                occ[uid].append((did, cid, txt))
        if ln % 2000 == 0:
            print(f"  pass2 {ln} van ban, {time.time()-t0:.0f}s", flush=True)

def toks(t):
    return set(t.lower().split())

def jac(a, b):
    A, B = toks(a), toks(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

same_dieu = same_doc_diff_dieu = diff_doc = 0
identical = different = 0
jacs = []
worst = []
for uid, lst in occ.items():
    dids = {d for d, _c, _t in lst}
    cids = {c for _d, c, _t in lst}
    if len(cids) > 1:
        diff_doc += 1
    elif len(dids) > 1:
        same_doc_diff_dieu += 1
    else:
        same_dieu += 1
    first_txt, last_txt = lst[0][2], lst[-1][2]
    if first_txt.strip() == last_txt.strip():
        identical += 1
    else:
        different += 1
        jacs.append(jac(first_txt, last_txt))
    worst.append((len(lst), uid))

n_dup = len(occ)
print(f"\n=== 1. BAN SAO NAM O DAU (n={n_dup} ma trung) ===")
print(f"Cung mot Dieu                : {same_dieu:6d} = {same_dieu/n_dup:5.1%}  (thiet hai NHO)")
print(f"Khac Dieu, cung van ban      : {same_doc_diff_dieu:6d} = {same_doc_diff_dieu/n_dup:5.1%}")
print(f"KHAC VAN BAN                 : {diff_doc:6d} = {diff_doc/n_dup:5.1%}  (thiet hai LON)")

print(f"\n=== 2. TEXT: hang GIAO C (dau) vs hang CHAM DIEM (cuoi) ===")
print(f"Giong het (vo hai)           : {identical:6d} = {identical/n_dup:5.1%}")
print(f"Khac nhau (co hai)           : {different:6d} = {different/n_dup:5.1%}")
if jacs:
    jacs.sort()
    print(f"  Jaccard text lech nhau: median {statistics.median(jacs):.3f}  "
          f"mean {statistics.fmean(jacs):.3f}  p10 {jacs[len(jacs)//10]:.3f}  "
          f"p90 {jacs[9*len(jacs)//10]:.3f}")
    print(f"  Hoan toan roi nhau (Jaccard=0): {sum(1 for j in jacs if j==0)} "
          f"= {sum(1 for j in jacs if j==0)/len(jacs):.1%} so ma co hai")

worst.sort(reverse=True)
print(f"\n=== 3. MUOI MA NHIEU BAN SAO NHAT ===")
for cnt, uid in worst[:10]:
    lst = occ[uid]
    n_cid = len({c for _d, c, _t in lst})
    print(f"  {uid:<22} x{cnt:<5} trai tren {n_cid} van ban")

print(f"\nTong thoi gian: {time.time()-t0:.0f}s")
