"""
PRIVATE — hybrid + rerank cho 1.918 cau private-official.json. MOT NHANH.

SOAN SAN, CHUA CHAY. Doi LB cua luot dau ve roi moi day — private khong do duoc
gi (0/1.918 co dap an) nen no la luot chay SAN XUAT, chay tren cau hinh co the
bi thay la phi GPU.

=============================================================================
HANG SO DUY NHAT CAN SUA KHI LB VE — MOT DONG
=============================================================================
"""
ARM = "title"          # <-- SUA O DAY. "title" hoac "bare". Nhanh thang do public quyet.
"""
=============================================================================

ALPHA / MAX_CONTEXTS / PER_ITEM / MAX_TITLE_SYLLABLES **KHONG** nam o day.
Chung duoc ap o buoc BUILD duoi may, va da la tham so dong lenh:

    python pipeline/build_qa_packages_public_v8_tieu_de_khoan.py \
        --rerank-path outputs/layer3/L3_rerank_private_title.jsonl \
        --alpha 0.7 --max-contexts 2 --per-item 200 \
        --expected-median 450 --out outputs/qa_packages_private_a0.7_k2_b200.json

    (--out BAT BUOC: resolve_out_path suy ten tu --rerank-path nen bo trong van
     ra ten khac, nhung dat ro rang de khong lan voi ban public)

MOT NHANH, khong phai hai. Private khong cham duoc nen chay hai nhanh cho ra hai
bo ket qua ma khong co cach nao chon giua chung.

KHAC ban public o DUNG MOT CHO: phai TU NHUNG 1.918 cau hoi private. Kernel E4
nhung qa_train / qa_public / retrieve_train — luc do private chua ton tai.

THU TU O LENH — O 1 LA CHOT CHAN, DUNG DAO:
  1. Dung src/b2_retrieval/ roi pickle.load index BM25   <- chet o day thi chet
                                                            trong 1 phut
  2. Nap nhung corpus + kiem 432.473 hang khop unit_index.tsv
  3. Nap private-official.json + NHUNG 1.918 cau hoi
  4. CHAY THU 20 CAU — bat loi duong dan/schema/mount trong 2 phut thay vi phut 70
  5. Ro hybrid top-50 cho du 1.918 cau
  6. Rerank day du, fp16
  7. Xuat jsonl + md5
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

t_start = time.time()

TOP_K_DOCS_LAYER1 = 100
DENSE_TOP_M = 300
TOP_K_OUT = 50
TRIAL_N = 20
MODEL_EMBED = "AITeamVN/Vietnamese_Embedding"
MODEL_RERANK = "AITeamVN/Vietnamese_Reranker"
MAX_SEQ_LEN = 1024
BATCH_SIZE = 256
RERANK_BATCH = 128

# ---------------------------------------------------------------------------
# O 1 — CHOT CHAN
# ---------------------------------------------------------------------------
print("=" * 70, flush=True)
print(f"O 1 — src/b2_retrieval/ + pickle.load BM25 (CHOT CHAN) | ARM={ARM}", flush=True)
print("=" * 70, flush=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "sympy"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rank_bm25", "underthesea"],
               check=False)

SRC = Path("/kaggle/working/src")
(SRC / "b2_retrieval").mkdir(parents=True, exist_ok=True)
(SRC / "__init__.py").write_text("", encoding="utf-8")
(SRC / "b2_retrieval" / "__init__.py").write_text("", encoding="utf-8")
found = list(Path("/kaggle/input").rglob("retrieve.py"))
assert found, "Khong thay retrieve.py (dataset dsc-legalqa-retrieval-src)"
(SRC / "b2_retrieval" / "retrieve.py").write_text(found[0].read_text(encoding="utf-8"),
                                                  encoding="utf-8")
sys.path.insert(0, "/kaggle/working")
from src.b2_retrieval import retrieve  # noqa: E402

bm25_found = list(Path("/kaggle/input").rglob("bm25_doc_index.pkl"))
assert bm25_found, "Khong thay bm25_doc_index.pkl"
t0 = time.time()
bm25_index = retrieve.load_index(bm25_found[0])
print(f"  BM25 index: {len(bm25_index.context_ids)} van ban ({time.time()-t0:.1f}s)", flush=True)
assert len(bm25_index.context_ids) > 8000, "Index qua nho — sai file?"
print("  O 1 QUA.\n", flush=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)

# ---------------------------------------------------------------------------
# O 2 — nhung corpus
# ---------------------------------------------------------------------------
print("=" * 70, flush=True)
print("O 2 — nap nhung corpus + kiem 432.473 hang", flush=True)
print("=" * 70, flush=True)

idx_found = list(Path("/kaggle/input").rglob("unit_index.tsv"))
assert idx_found, "Khong thay unit_index.tsv (output kernel E4)"
corpus_unit_ids = []
with open(idx_found[0], encoding="utf-8") as f:
    next(f)
    for line in f:
        corpus_unit_ids.append(line.rstrip("\n").split("\t")[3])
N_UNITS = len(corpus_unit_ids)
assert N_UNITS == 432473, f"LECH: mong 432473, thay {N_UNITS} — dung lai"
print(f"  unit_index.tsv: {N_UNITS} hang  OK", flush=True)

emb_p = list(Path("/kaggle/input").rglob(f"corpus_embeddings_{ARM}.npy"))
assert emb_p, f"Khong thay corpus_embeddings_{ARM}.npy (output kernel E4)"
corpus_emb = np.load(emb_p[0]).astype(np.float32)
assert corpus_emb.shape[0] == N_UNITS, "Nhung lech so hang so voi unit_index.tsv"
print(f"  corpus_embeddings_{ARM}: {corpus_emb.shape}", flush=True)
emb_index = retrieve.build_corpus_embedding_index(corpus_unit_ids)
UNIT_IDS_MD5 = hashlib.md5("\n".join(sorted(corpus_unit_ids)).encode("utf-8")).hexdigest()
print(f"  md5(unit_id da sap) = {UNIT_IDS_MD5}", flush=True)
print("  O 2 QUA.\n", flush=True)

# ---------------------------------------------------------------------------
# O 3 — nap private + NHUNG cau hoi
# ---------------------------------------------------------------------------
print("=" * 70, flush=True)
print("O 3 — nap private-official.json + nhung 1.918 cau hoi", flush=True)
print("=" * 70, flush=True)

pc_found = list(Path("/kaggle/input").rglob("parsed_corpus.jsonl"))
pv_found = list(Path("/kaggle/input").rglob("private-official.json"))
assert pc_found, "Khong thay parsed_corpus.jsonl"
assert pv_found, ("Khong thay private-official.json — da them vao dataset chua? "
                  "Nho: `datasets version` tao PHIEN BAN MOI, kernel ghim ban cu se "
                  "khong thay file moi.")

t0 = time.time()
parsed_corpus = {}
with open(pc_found[0], encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        parsed_corpus[d["context_id"]] = d
qa = json.loads(pv_found[0].read_text(encoding="utf-8"))
print(f"  parsed_corpus {len(parsed_corpus)} van ban | private {len(qa)} cau "
      f"({time.time()-t0:.1f}s)", flush=True)
assert len(qa) == 1918, f"LECH: mong 1918 cau private, thay {len(qa)}"

# BAY IM LANG: phai nhung cau hoi bang DUNG cau hinh da nhung kho. Lech thi
# KHONG BAO LOI, chi la truy hoi te di va khong ai biet vi sao. In ra thanh
# khang dinh, dung de ngam.
from sentence_transformers import SentenceTransformer  # noqa: E402

emb_model = SentenceTransformer(MODEL_EMBED, device=DEVICE)
emb_model.max_seq_length = MAX_SEQ_LEN
if DEVICE == "cuda":
    emb_model.half()
print("\n  --- KHOP CAU HINH NHUNG VOI KHO (kernel E4) ---", flush=True)
print(f"  model                 = {MODEL_EMBED}            [BAT BUOC khop]", flush=True)
print(f"  normalize_embeddings  = True                     [BAT BUOC khop — lech thi", flush=True)
print( "                                                    tich vo huong khong con la", flush=True)
print( "                                                    cosine, thu hang SAI HET]", flush=True)
print(f"  model.half()          = {DEVICE == 'cuda'}                     [nen khop]", flush=True)
print(f"  max_seq_length        = {MAX_SEQ_LEN}                     [khong quan trong:", flush=True)
print( "                                                    cau hoi median 19 am tiet]", flush=True)

qids = list(qa.keys())
t0 = time.time()
q_emb = emb_model.encode([qa[q]["question"] for q in qids], batch_size=BATCH_SIZE,
                         show_progress_bar=True, normalize_embeddings=True,
                         convert_to_numpy=True).astype(np.float32)
print(f"\n  nhung {len(qids)} cau hoi: {time.time()-t0:.1f}s  shape={q_emb.shape}", flush=True)
qid_to_qemb = dict(zip(qids, q_emb))
del emb_model
if DEVICE == "cuda":
    torch.cuda.empty_cache()
print("  O 3 QUA.\n", flush=True)


def build_basket(qid_list, label):
    out = {}
    t = time.time()
    for i, qid in enumerate(qid_list):
        out[qid] = retrieve.search_units_hybrid(
            bm25_index, qa[qid]["question"], parsed_corpus,
            corpus_emb, corpus_unit_ids, emb_index, qid_to_qemb[qid],
            top_k_docs=TOP_K_DOCS_LAYER1, dense_top_m=DENSE_TOP_M, top_k_out=TOP_K_OUT,
        )
        if (i + 1) % 400 == 0:
            print(f"    {label} {i+1}/{len(qid_list)}  {time.time()-t:.0f}s", flush=True)
    return out, time.time() - t


# ---------------------------------------------------------------------------
# O 4 — CHAY THU 20 CAU
# ---------------------------------------------------------------------------
print("=" * 70, flush=True)
print(f"O 4 — CHAY THU {TRIAL_N} CAU (bat loi trong 2 phut, khong phai phut 70)", flush=True)
print("=" * 70, flush=True)

trial_qids = qids[:TRIAL_N]
trial, dt = build_basket(trial_qids, "thu")
sizes = [len(v) for v in trial.values()]
n_empty = sum(1 for us in trial.values() for u in us if not u["text"].strip())
per_q = dt / TRIAL_N
print(f"  {TRIAL_N} cau trong {dt:.1f}s = {per_q:.2f}s/cau", flush=True)
print(f"  -> ngoai suy {len(qids)} cau = {per_q*len(qids)/60:.1f} phut", flush=True)
print(f"  KIEM: ung vien/cau min={min(sizes)} max={max(sizes)} (ky vong {TOP_K_OUT})", flush=True)
print(f"  KIEM: candidate text rong = {n_empty} (ky vong 0)", flush=True)
assert min(sizes) == max(sizes) == TOP_K_OUT, "So ung vien lech — dung lai"
assert n_empty == 0, "Co candidate text rong — dung lai"
print("  O 4 QUA.\n", flush=True)

# ---------------------------------------------------------------------------
# O 5 — ro hybrid day du
# ---------------------------------------------------------------------------
print("=" * 70, flush=True)
print(f"O 5 — ro hybrid top-{TOP_K_OUT} cho {len(qids)} cau", flush=True)
print("=" * 70, flush=True)
baskets, dt = build_basket(qids, "day du")
sizes = [len(v) for v in baskets.values()]
n_empty = sum(1 for us in baskets.values() for u in us if not u["text"].strip())
print(f"  xong {dt:.1f}s | ung vien/cau min={min(sizes)} max={max(sizes)} | "
      f"text rong={n_empty}", flush=True)
assert min(sizes) == max(sizes) == TOP_K_OUT and n_empty == 0, "Kiem that bai — dung lai"
print("  O 5 QUA.\n", flush=True)

# ---------------------------------------------------------------------------
# O 6 — rerank fp16
# ---------------------------------------------------------------------------
from sentence_transformers import CrossEncoder  # noqa: E402

print("=" * 70, flush=True)
print("O 6 — rerank fp16", flush=True)
print("=" * 70, flush=True)
model = CrossEncoder(MODEL_RERANK, max_length=512, device=DEVICE)
if DEVICE == "cuda":
    model.model.half()
    print("  fp16 BAT (T3 do 147 cap/giay, gap 3,2 lan con so 46 cua Task 1)", flush=True)

pairs, spans = [], []
for qid in qids:
    us = baskets[qid]
    spans.append((qid, len(pairs), len(us)))
    for u in us:
        pairs.append([qa[qid]["question"], u["text"]])
t0 = time.time()
scores = model.predict(pairs, batch_size=RERANK_BATCH, show_progress_bar=True)
dt = time.time() - t0
print(f"  {len(pairs)} cap trong {dt:.1f}s = {len(pairs)/dt:.1f} cap/giay", flush=True)

OUT_DIR = Path("/kaggle/working/layer3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / f"L3_rerank_private_{ARM}.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for qid, start, n in spans:
        us = baskets[qid]
        scored = [{
            "unit_id": u["unit_id"],
            "context_id": u["context_id"],
            "row": int(u.get("row", -1)),
            "score": float(scores[start + j]),
            "dense_rank": j,
        } for j, u in enumerate(us)]
        scored.sort(key=lambda x: -x["score"])
        f.write(json.dumps({"qid": qid, "question": qa[qid]["question"],
                            "ranked_units": scored}, ensure_ascii=False) + "\n")
print(f"  -> {out_path.name} ({out_path.stat().st_size/1024/1024:.1f} MB)", flush=True)
print("  O 6 QUA.\n", flush=True)

# ---------------------------------------------------------------------------
# O 7 — md5
# ---------------------------------------------------------------------------
h = hashlib.md5()
with open(out_path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
print("=" * 70, flush=True)
print(f"  md5(unit_id da sap)      = {UNIT_IDS_MD5}", flush=True)
print(f"  md5({out_path.name}) = {h.hexdigest()}", flush=True)
(OUT_DIR / "MD5SUMS.txt").write_text(
    f"unit_ids_sorted {UNIT_IDS_MD5}\n{out_path.name} {h.hexdigest()}\n", encoding="utf-8")

import shutil  # noqa: E402

shutil.make_archive("/kaggle/working/private_rerank_results", "zip", OUT_DIR)
print(f"\nXONG. Tong {(time.time()-t_start)/60:.1f} phut.", flush=True)
print("Tai /kaggle/working/private_rerank_results.zip", flush=True)
