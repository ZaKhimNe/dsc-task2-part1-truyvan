"""
T3 — Rerank Layer 3 tren RO HYBRID, hai nhanh (bare / title), MOT LAN CHAY.

MUC TIEU: ra `L3_rerank_public_bare.jsonl` va `L3_rerank_public_title.jsonl`,
cung mot lan chay, cung duong code. KHONG build goi V8/V9 (phien khac lo).

VI SAO PHAI CUNG MOT LAN CHAY: lay nhanh bare tu lan chay khac la tai hien bay H4
cua Task 1 — hai artefact mo ta hai cai ro ma ten file khong bao.

BIEN DUY NHAT: bo nhung dense (corpus_embeddings_bare vs _title). Nhanh BM25 goi
cung mot ham, cung tham so, tren cung index -> deterministic, ra y het nhau.

VI SAO GIO MOI DUNG LUC: Task 1 do reranker tren ro yeu ra +0, tren ro tot ra +1,76.
Ro Task 2 dang yeu; E4 vua lam no tot hon (+4,40 diem "voi toi duoc trong top-50").

THU TU O LENH — O 1 LA CHOT CHAN, DUNG DAO:
  1. Dung src/b2_retrieval/ roi pickle.load index BM25   <- chet o day thi chet
                                                            trong 1 phut, khong
                                                            phai phut 38
  2. Nap hai bo nhung + kiem 432.473 hang khop unit_index.tsv
  3. Nap qa_public
  4. Dung ro hybrid top-50 ca hai nhanh (co do thu 20 cau + ngoai suy truoc)
  5. Chay thu rerank 50 cau, do toc do that, in ngoai suy
  6. Rerank day du, BAT fp16
  7. Xuat 2 jsonl + md5 tung file + md5 danh sach unit_id da sap

fp16: rerank_notebook.py dong 30 hien chi co CrossEncoder(MODEL_NAME, max_length=512,
device=DEVICE) — KHONG co .half(). Task 1 do 11 -> 46 cap/giay, da kiem an toan:
top-5 khop 10/10, lech diem lon nhat 0,0014.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

t_start = time.time()

# ---------------------------------------------------------------------------
# O 1 — CHOT CHAN: dung cau truc module roi nap index BM25
# ---------------------------------------------------------------------------
print("=" * 70, flush=True)
print("O 1 — dung src/b2_retrieval/ + nap index BM25 (CHOT CHAN)", flush=True)
print("=" * 70, flush=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "sympy"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rank_bm25", "underthesea"],
               check=False)

SRC = Path("/kaggle/working/src")
(SRC / "b2_retrieval").mkdir(parents=True, exist_ok=True)
(SRC / "__init__.py").write_text("", encoding="utf-8")
(SRC / "b2_retrieval" / "__init__.py").write_text("", encoding="utf-8")

found = list(Path("/kaggle/input").rglob("retrieve.py"))
assert found, "Khong thay retrieve.py trong /kaggle/input (dataset dsc-legalqa-retrieval-src)"
(SRC / "b2_retrieval" / "retrieve.py").write_text(
    found[0].read_text(encoding="utf-8"), encoding="utf-8"
)
sys.path.insert(0, "/kaggle/working")
print(f"  retrieve.py <- {found[0]}", flush=True)

from src.b2_retrieval import retrieve  # noqa: E402

# pickle BM25 luu duong module `src.b2_retrieval.retrieve.BM25Index` — thieu cau truc
# tren la pickle.load chet ngay. Day la ly do o 1 phai chay truoc moi thu.
bm25_found = list(Path("/kaggle/input").rglob("bm25_doc_index.pkl"))
assert bm25_found, "Khong thay bm25_doc_index.pkl (dataset dsc-legalqa-bm25-index)"
t0 = time.time()
bm25_index = retrieve.load_index(bm25_found[0])
print(f"  BM25 index <- {bm25_found[0]}  ({time.time()-t0:.1f}s)", flush=True)
print(f"  So van ban trong index: {len(bm25_index.context_ids)}", flush=True)
assert len(bm25_index.context_ids) > 8000, "Index qua nho — sai file?"
print("  O 1 QUA.\n", flush=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)

# ---------------------------------------------------------------------------
# O 2 — nap hai bo nhung + kiem khop unit_index.tsv
# ---------------------------------------------------------------------------
print("=" * 70, flush=True)
print("O 2 — nap hai bo nhung + kiem 432.473 hang", flush=True)
print("=" * 70, flush=True)

idx_found = list(Path("/kaggle/input").rglob("unit_index.tsv"))
assert idx_found, "Khong thay unit_index.tsv (output kernel E4)"
corpus_unit_ids = []
with open(idx_found[0], encoding="utf-8") as f:
    next(f)
    for line in f:
        corpus_unit_ids.append(line.rstrip("\n").split("\t")[3])
N_UNITS = len(corpus_unit_ids)
print(f"  unit_index.tsv <- {idx_found[0]}", flush=True)
print(f"  So hang: {N_UNITS}", flush=True)
assert N_UNITS == 432473, f"LECH: mong 432473, thay {N_UNITS} — dung lai, dung chay tiep"

emb_paths = {}
for arm in ("bare", "title"):
    p = list(Path("/kaggle/input").rglob(f"corpus_embeddings_{arm}.npy"))
    assert p, f"Khong thay corpus_embeddings_{arm}.npy (output kernel E4)"
    emb_paths[arm] = p[0]
    print(f"  {arm:5s} <- {p[0]}", flush=True)

qemb_p = list(Path("/kaggle/input").rglob("query_embeddings_qa_public.npy"))
qids_p = list(Path("/kaggle/input").rglob("query_qids_qa_public.json"))
assert qemb_p and qids_p, "Khong thay query embedding qa_public (output kernel E4)"
q_emb = np.load(qemb_p[0]).astype(np.float32)
q_qids = json.loads(qids_p[0].read_text(encoding="utf-8"))
qid_to_qemb = dict(zip(q_qids, q_emb))
print(f"  query qa_public: {len(q_qids)} cau", flush=True)

# md5 danh sach unit_id da sap — bang chung hai nhanh mo ta CUNG mot kho
UNIT_IDS_MD5 = hashlib.md5("\n".join(sorted(corpus_unit_ids)).encode("utf-8")).hexdigest()
print(f"  md5(unit_id da sap) = {UNIT_IDS_MD5}", flush=True)
print("  O 2 QUA.\n", flush=True)

# ---------------------------------------------------------------------------
# O 3 — nap qa_public + parsed_corpus
# ---------------------------------------------------------------------------
print("=" * 70, flush=True)
print("O 3 — nap qa_public + parsed_corpus", flush=True)
print("=" * 70, flush=True)

pc_found = list(Path("/kaggle/input").rglob("parsed_corpus.jsonl"))
qp_found = list(Path("/kaggle/input").rglob("qa_public.json"))
assert pc_found and qp_found, "Thieu parsed_corpus.jsonl hoac qa_public.json"

t0 = time.time()
parsed_corpus = {}
with open(pc_found[0], encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        parsed_corpus[d["context_id"]] = d
qa_public = json.loads(qp_found[0].read_text(encoding="utf-8"))
print(f"  parsed_corpus {len(parsed_corpus)} van ban, qa_public {len(qa_public)} cau"
      f"  ({time.time()-t0:.1f}s)", flush=True)
assert len(qa_public) == 1000, f"LECH: mong 1000 cau, thay {len(qa_public)}"
print("  O 3 QUA.\n", flush=True)

# ---------------------------------------------------------------------------
# O 4 — dung ro hybrid top-50, ca hai nhanh
# ---------------------------------------------------------------------------
TOP_K_DOCS_LAYER1 = 100
DENSE_TOP_M = 300
TOP_K_OUT = 50

print("=" * 70, flush=True)
print(f"O 4 — ro hybrid top-{TOP_K_OUT} (top_k_docs={TOP_K_DOCS_LAYER1}, "
      f"dense_top_m={DENSE_TOP_M})", flush=True)
print("=" * 70, flush=True)

qids_all = [q for q in qa_public if q in qid_to_qemb]
print(f"  {len(qids_all)} cau co ca cau hoi va query embedding", flush=True)

baskets = {}
for arm in ("bare", "title"):
    print(f"\n  --- nhanh {arm} ---", flush=True)
    t0 = time.time()
    corpus_emb = np.load(emb_paths[arm]).astype(np.float32)
    print(f"  nap nhung: {corpus_emb.shape}  ({time.time()-t0:.1f}s)", flush=True)
    assert corpus_emb.shape[0] == N_UNITS, "Nhung lech so hang so voi unit_index.tsv"
    emb_index = retrieve.build_corpus_embedding_index(corpus_unit_ids)

    arm_baskets = {}
    # do thu 20 cau roi ngoai suy TRUOC khi chay het — khong de phat hien cham o phut 38
    t0 = time.time()
    for i, qid in enumerate(qids_all):
        units = retrieve.search_units_hybrid(
            bm25_index, qa_public[qid]["question"], parsed_corpus,
            corpus_emb, corpus_unit_ids, emb_index, qid_to_qemb[qid],
            top_k_docs=TOP_K_DOCS_LAYER1, dense_top_m=DENSE_TOP_M, top_k_out=TOP_K_OUT,
        )
        arm_baskets[qid] = units
        if i + 1 == 20:
            per_q = (time.time() - t0) / 20
            print(f"  [do thu] 20 cau trong {time.time()-t0:.1f}s = {per_q:.2f}s/cau"
                  f"  -> ngoai suy {len(qids_all)} cau = {per_q*len(qids_all)/60:.1f} phut"
                  f"  (ca 2 nhanh ~{per_q*len(qids_all)*2/60:.1f} phut)", flush=True)
        if (i + 1) % 250 == 0:
            print(f"    {i+1}/{len(qids_all)}  {time.time()-t0:.0f}s", flush=True)
    print(f"  nhanh {arm} xong: {time.time()-t0:.1f}s", flush=True)

    n_empty = sum(1 for us in arm_baskets.values() for u in us if not u["text"].strip())
    sizes = [len(us) for us in arm_baskets.values()]
    print(f"  KIEM: candidate text rong = {n_empty} (ky vong 0)", flush=True)
    print(f"  KIEM: so ung vien/cau min={min(sizes)} max={max(sizes)} (ky vong 50)", flush=True)
    baskets[arm] = arm_baskets

    del corpus_emb, emb_index
    import gc
    gc.collect()

print("  O 4 QUA.\n", flush=True)

# ---------------------------------------------------------------------------
# O 5 — chay thu rerank 50 cau, do toc do that
# ---------------------------------------------------------------------------
from sentence_transformers import CrossEncoder  # noqa: E402

MODEL_NAME = "AITeamVN/Vietnamese_Reranker"
MAX_LENGTH = 512
BATCH_SIZE = 128

print("=" * 70, flush=True)
print("O 5 — chay thu 50 cau, do toc do THAT", flush=True)
print("=" * 70, flush=True)
print("  Task 1 sai hai lan cung nguyen nhan: uoc 1h hoa 6,9h; uoc 1,2h hoa 5,4h.", flush=True)

t0 = time.time()
model = CrossEncoder(MODEL_NAME, max_length=MAX_LENGTH, device=DEVICE)
if DEVICE == "cuda":
    model.model.half()   # fp16 — Task 1 do 11 -> 46 cap/giay
    print("  fp16 BAT", flush=True)
print(f"  nap model: {time.time()-t0:.1f}s", flush=True)

trial_qids = qids_all[:50]
trial_pairs = []
for qid in trial_qids:
    q = qa_public[qid]["question"]
    for u in baskets["bare"][qid]:
        trial_pairs.append([q, u["text"]])
t0 = time.time()
_ = model.predict(trial_pairs, batch_size=BATCH_SIZE, show_progress_bar=False)
dt = time.time() - t0
rate = len(trial_pairs) / dt
total_pairs = sum(len(us) for arm in baskets for us in baskets[arm].values())
print(f"  50 cau = {len(trial_pairs)} cap trong {dt:.1f}s = {rate:.1f} cap/giay", flush=True)
print(f"  TONG phai cham: {total_pairs} cap -> ngoai suy {total_pairs/rate/60:.1f} phut", flush=True)
print("  O 5 QUA.\n", flush=True)

# ---------------------------------------------------------------------------
# O 6 — rerank day du
# ---------------------------------------------------------------------------
print("=" * 70, flush=True)
print("O 6 — rerank day du, fp16", flush=True)
print("=" * 70, flush=True)

OUT_DIR = Path("/kaggle/working/layer3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
results = {}

for arm in ("bare", "title"):
    print(f"\n  --- rerank nhanh {arm} ---", flush=True)
    t0 = time.time()
    pairs = []
    spans = []
    for qid in qids_all:
        q = qa_public[qid]["question"]
        us = baskets[arm][qid]
        spans.append((qid, len(pairs), len(us)))
        for u in us:
            pairs.append([q, u["text"]])
    scores = model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=True)
    dt = time.time() - t0
    print(f"  {len(pairs)} cap trong {dt:.1f}s = {len(pairs)/dt:.1f} cap/giay", flush=True)

    out_path = OUT_DIR / f"L3_rerank_public_{arm}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for qid, start, n in spans:
            us = baskets[arm][qid]
            scored = []
            for j, u in enumerate(us):
                scored.append({
                    "unit_id": u["unit_id"],
                    "context_id": u["context_id"],
                    "row": int(u.get("row", -1)),
                    "score": float(scores[start + j]),
                    "dense_rank": j,
                })
            scored.sort(key=lambda x: -x["score"])
            f.write(json.dumps({"qid": qid, "question": qa_public[qid]["question"],
                                "ranked_units": scored}, ensure_ascii=False) + "\n")
    results[arm] = out_path
    print(f"  -> {out_path.name}  ({out_path.stat().st_size/1024/1024:.1f} MB)", flush=True)

print("  O 6 QUA.\n", flush=True)

# ---------------------------------------------------------------------------
# O 7 — xuat + md5
# ---------------------------------------------------------------------------
print("=" * 70, flush=True)
print("O 7 — md5", flush=True)
print("=" * 70, flush=True)


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


print(f"\n  md5(unit_id da sap)  = {UNIT_IDS_MD5}", flush=True)
print("  (hai nhanh dung chung gia tri nay -> cung mot kho, cung thu tu hang)\n", flush=True)
for arm, p in results.items():
    print(f"  md5({p.name}) = {md5_file(p)}", flush=True)

(OUT_DIR / "MD5SUMS.txt").write_text(
    f"unit_ids_sorted {UNIT_IDS_MD5}\n"
    + "".join(f"{p.name} {md5_file(p)}\n" for p in results.values()),
    encoding="utf-8",
)

import shutil  # noqa: E402

shutil.make_archive("/kaggle/working/hybrid_rerank_results", "zip", OUT_DIR)
print(f"\nXONG. Tong {(time.time()-t_start)/60:.1f} phut.", flush=True)
print("Tai /kaggle/working/hybrid_rerank_results.zip", flush=True)
