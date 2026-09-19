"""
E4 — Nhúng corpus KÈM TIÊU ĐỀ ĐIỀU, chạy A/B trong CÙNG một phiên GPU.
(P3 của PLAN_TOI_THUONG_RETRIEVAL_v2.md §4)

GIẢ THUYẾT: bi-encoder đang mã hoá 50,3% unit dưới 50 âm tiết mà không có chủ
đề. Thêm dòng `dieu_tieu_de` vào trước text lúc nhúng sẽ kéo thêm gold vào rổ,
nhất là nhóm "đúng văn bản sai Khoản".

VÌ SAO MỘT PHIÊN, HAI NHÁNH: lần chạy 18/09 07:18 dừng ở cell 6, nên baseline
cũng CHƯA có embedding. Phải sinh cả hai nhánh mới so được. Nhúng chung một
phiên còn đảm bảo cùng model, cùng seed, cùng phiên bản thư viện — đúng nguyên
tắc "đổi một biến mỗi lần" (plan §0 quy tắc 2).

ĐỔI ĐÚNG MỘT BIẾN so với embed_corpus_notebook.py:
    bare  : k["text"]                            <- nhánh A, y hệt bản gốc
    title : dieu_tieu_de + "\\n" + k["text"]      <- nhánh B
Mọi thứ khác giữ nguyên: MODEL_NAME, max_seq_length=1024, model.half(),
batch_size=256, normalize_embeddings=True, và THỨ TỰ DUYỆT corpus.

BA CHẶN KHI GHÉP TIÊU ĐỀ (đo trên corpus thật 18/09, xem src/b6_context_package/
format_unit.py): 1,1% Điều không có tiêu đề; 5,6% Điều có text == dieu_tieu_de
(ghép vào sẽ thành dòng lặp); tiêu đề median 9 âm tiết. Cả hai case đầu -> thoái
về text trần, để nhánh B không bao giờ tệ hơn nhánh A vì lý do định dạng.

XUẤT GÌ: KHÔNG tải embedding 800MB về máy. Chạy luôn bước search trên Kaggle,
chỉ xuất top-50 mỗi câu cho từng nhánh (vài chục MB) — đúng "Cách B" mà
kaggle_layer2/README.md khuyên sau vụ tải đứt 13/08.

Xuất kèm ROW INDEX cạnh unit_id. Đây là cách vá bug `build_corpus_embedding_index`
(retrieve.py:413 — mã trùng thì bản sau đè bản trước, che 38.475 hàng = 8,9%):
có vị trí hàng thì tra đúng đoạn ĐÃ KHỚP, không phải đoạn cuối cùng mang mã đó.

Embedding vẫn được lưu trong /kaggle/working để phiên sau mount lại làm
kernel_sources, khỏi phải nhúng lại 50 phút GPU.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

# sympy cua image Kaggle xung dot voi sentence_transformers — nang cap TRUOC moi
# import torch/ST. Trong batch script chua co gi import sympy nen khong can restart.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "sympy"], check=False)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

# --- CELL 3 tuong duong: tim dataset, khong doan tien to duong dan -----------
found = list(Path("/kaggle/input").rglob("parsed_corpus.jsonl"))
assert found, "Khong thay parsed_corpus.jsonl trong /kaggle/input."
INPUT_DIR = found[0].parent
print("INPUT_DIR:", INPUT_DIR, flush=True)

OUT_DIR = Path("/kaggle/working/layer2_ab")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE, flush=True)

MODEL_NAME = "AITeamVN/Vietnamese_Embedding"
BATCH_SIZE = 256
MAX_SEQ_LEN = 1024
TOP_K = 50
QUERY_SOURCES = {
    "qa_train": "qa_train.json",
    "qa_public": "qa_public.json",
    "retrieve_train": "data_retrieve_train.json",
}

# --- CELL 5 tuong duong: doc corpus, dung HAI bien the text -----------------
# THU TU DUYET PHAI GIONG HET ban goc — row index cua ca hai nhanh va cua moi
# lan chay sau deu phai khop nhau.
rows_meta = []      # (context_id, dieu_id, unit_id)
texts_bare = []
texts_title = []
n_no_title = 0
n_text_eq_title = 0

t0 = time.time()
with open(INPUT_DIR / "parsed_corpus.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        cid = r["context_id"]
        if r["parse_status"] == "fallback":
            if r["dieu"]:
                d0 = r["dieu"][0]
                uid = d0["dieu_id"] or cid
                rows_meta.append((cid, d0.get("dieu_id", ""), uid))
                texts_bare.append(d0["text"])
                texts_title.append(d0["text"])   # fallback: khong co tieu de rieng
                n_no_title += 1
            continue
        for dieu in r["dieu"]:
            td = (dieu.get("dieu_tieu_de") or "").strip()
            dtext = (dieu.get("text") or "").strip()
            usable_title = bool(td) and dtext != td
            if not td:
                n_no_title += 1
            elif dtext == td:
                n_text_eq_title += 1
            if dieu["khoan"]:
                for k in dieu["khoan"]:
                    body = k["text"]
                    rows_meta.append((cid, dieu.get("dieu_id", ""), k["khoan_id"]))
                    texts_bare.append(body)
                    texts_title.append(f"{td}\n{body}" if usable_title else body)
            else:
                body = dieu["text"]
                rows_meta.append((cid, dieu.get("dieu_id", ""), dieu["dieu_id"]))
                texts_bare.append(body)
                texts_title.append(f"{td}\n{body}" if usable_title else body)

N = len(rows_meta)
print(f"Doc xong {N} unit trong {time.time()-t0:.1f}s", flush=True)
assert len(texts_bare) == len(texts_title) == N
n_changed = sum(1 for a, b in zip(texts_bare, texts_title) if a != b)
print(f"  Dieu khong co tieu de        : {n_no_title}", flush=True)
print(f"  Dieu co text == dieu_tieu_de : {n_text_eq_title}", flush=True)
print(f"  Hang THUC SU doi text        : {n_changed} = {n_changed/N:.1%}", flush=True)

# Bang tra row -> id, de o may local map nguoc ma KHONG qua bang tra bi bug
with open(OUT_DIR / "unit_index.tsv", "w", encoding="utf-8") as f:
    f.write("row\tcontext_id\tdieu_id\tunit_id\n")
    for i, (cid, did, uid) in enumerate(rows_meta):
        f.write(f"{i}\t{cid}\t{did}\t{uid}\n")
print("Da ghi unit_index.tsv", flush=True)

# --- Model -----------------------------------------------------------------
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
model.max_seq_length = MAX_SEQ_LEN
if DEVICE == "cuda":
    model.half()
DIM = model.get_sentence_embedding_dimension()
print(f"Model loaded. dim={DIM}", flush=True)


def embed(texts, tag):
    t = time.time()
    emb = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"Embed {tag} xong: {time.time()-t:.1f}s shape={emb.shape}", flush=True)
    np.save(OUT_DIR / f"corpus_embeddings_{tag}.npy", emb.astype(np.float16))
    return emb


emb_bare = embed(texts_bare, "bare")
emb_title = embed(texts_title, "title")

# --- Query: nhung MOT LAN, dung chung cho ca hai nhanh ----------------------
# (cau hoi khong doi giua hai nhanh — chi corpus doi)
queries = {}
for tag, fname in QUERY_SOURCES.items():
    p = INPUT_DIR / fname
    if not p.exists():
        print(f"BO QUA {tag}: khong thay {p}", flush=True)
        continue
    data = json.loads(p.read_text(encoding="utf-8"))
    qids = list(data.keys())
    qtexts = [data[q]["question"] for q in qids]
    t = time.time()
    qemb = model.encode(qtexts, batch_size=BATCH_SIZE, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True)
    print(f"{tag}: {len(qids)} cau, {time.time()-t:.1f}s", flush=True)
    np.save(OUT_DIR / f"query_embeddings_{tag}.npy", qemb.astype(np.float16))
    (OUT_DIR / f"query_qids_{tag}.json").write_text(json.dumps(qids), encoding="utf-8")
    queries[tag] = (qids, qemb)

# --- Search tren GPU, xuat top-50 (row index + unit_id + score) -------------
def search_and_dump(corpus_emb, arm):
    ce = torch.from_numpy(corpus_emb).to(DEVICE)
    if DEVICE == "cuda":
        ce = ce.half()
    for tag, (qids, qemb) in queries.items():
        out = OUT_DIR / f"top{TOP_K}_{arm}_{tag}.jsonl"
        t = time.time()
        with open(out, "w", encoding="utf-8") as f:
            CH = 256
            for s in range(0, len(qids), CH):
                qb = torch.from_numpy(qemb[s:s + CH]).to(DEVICE)
                if DEVICE == "cuda":
                    qb = qb.half()
                scores = qb @ ce.T                       # (chunk, N)
                top = torch.topk(scores, k=TOP_K, dim=1)
                idx = top.indices.cpu().numpy()
                val = top.values.float().cpu().numpy()
                for j in range(idx.shape[0]):
                    rowsj = idx[j].tolist()
                    f.write(json.dumps({
                        "qid": qids[s + j],
                        "rows": rowsj,
                        "unit_ids": [rows_meta[r][2] for r in rowsj],
                        "context_ids": [rows_meta[r][0] for r in rowsj],
                        "scores": [round(float(v), 5) for v in val[j].tolist()],
                    }, ensure_ascii=False) + "\n")
        print(f"  {arm}/{tag}: {len(qids)} cau -> {out.name}  {time.time()-t:.1f}s", flush=True)
    del ce
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


print("Search nhanh BARE...", flush=True)
search_and_dump(emb_bare, "bare")
print("Search nhanh TITLE...", flush=True)
search_and_dump(emb_title, "title")

# --- Goi ket qua NHE de tai ve ---------------------------------------------
import shutil  # noqa: E402

light = Path("/kaggle/working/e4_results")
light.mkdir(exist_ok=True)
for p in OUT_DIR.iterdir():
    if p.suffix in (".jsonl", ".tsv") or p.name.startswith("query_qids"):
        shutil.copy(p, light / p.name)
shutil.make_archive("/kaggle/working/e4_results", "zip", light)
shutil.rmtree(light)
print("\nXONG. Tai /kaggle/working/e4_results.zip (nhe) —"
      " embedding .npy giu lai trong /kaggle/working/layer2_ab/ cho phien sau.", flush=True)
