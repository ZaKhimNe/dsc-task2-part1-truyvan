# P2 — Fine-tune bi-encoder o muc KHOAN, roi do tren tap GIU RIENG.
#
# Chong lung: Task 1 an +1,90 diem bang THAT tu dung viec nay o muc VAN BAN.
# Khac biet bat buoc o day: du lieu day la cap (cau hoi <-> KHOAN), va negative
# la Khoan KHAC trong CUNG DIEU - vi viec can hoc la TACH cac Khoan trong mot
# Dieu ra khoi nhau, thu ma negative ngau nhien khong bao gio day duoc.
#
# Ba loi da bat trong khau dung du lieu (xem pipeline/build_khoan_*.py):
#   1. chon nhan bang recall -> chon Khoan DAI NHAT (bay E6). Da doi sang F1.
#   2. thieu negative cung Dieu -> model chi hoc chu de. Da them, 35,6%.
#   3. positive/negative lech do dai 6 lan -> bo phan loai CHI DUNG DO DAI dat
#      94,3%. Da can bang do dai, con 68,6%, duoi muc TU NHIEN cua kho la 74,2%.
#
# NGUONG DAT TRUOC (bang nguong E4, la thi nghiem duy nhat tung vuot):
#   recall@50 tren tap giu rieng >= +2,0 diem  VA  can duoi CI95 > 0.
#   Khong dat -> KHONG dem ra san xuat, bao so va dung.
#
# Mount:
#   - dataset  zakhim/dsc2026-legalqa-khoan-triplets   (triplet)
#   - dataset  co parsed_corpus.jsonl
#   - kernel   zakhim/dsc-legalqa-e4-title-embedding   (embedding CU + unit_index.tsv)
# Accelerator: GPU T4 x2 · Internet: On

# %% CELL 1 — moi truong
import json, os, random, time
from pathlib import Path
import numpy as np
import torch

SEED = 2026
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device", DEVICE,
      "| gpu", torch.cuda.device_count(), flush=True)

# %% CELL 2 — tim input
def find(pattern, what):
    hits = sorted(Path("/kaggle/input").rglob(pattern))
    assert hits, f"Khong thay {what} ({pattern}) trong /kaggle/input"
    print(f"  {what}: {hits[0]}", flush=True)
    return hits[0]

P_TRAIN   = find("khoan_triplets_train.jsonl",   "triplet day")
P_HELD    = find("khoan_triplets_heldout.jsonl", "triplet giu rieng")
P_CORPUS  = find("parsed_corpus.jsonl",          "corpus")
P_OLD_EMB = find("corpus_embeddings_title.npy",  "embedding CU (E4)")
P_UIDX    = find("unit_index.tsv",               "bang tra row->unit_id")

OUT = Path("/kaggle/working/p2_finetune"); OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "AITeamVN/Vietnamese_Embedding"
MAX_SEQ_LEN = 1024
BATCH_SIZE_EMB = 256
TOP_K = 50

# %% CELL 3 — nap triplet
def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

tr = load_jsonl(P_TRAIN)
ho = load_jsonl(P_HELD)
print(f"day {len(tr)} triplet · giu rieng {len(ho)} triplet", flush=True)

# kiem ro ri: khong cau nao duoc o ca hai ben
ov = {r["qid"] for r in tr} & {r["qid"] for r in ho}
assert not ov, f"RO RI {len(ov)} cau nam o ca hai ben"
print("  kiem ro ri: 0 cau trung — OK", flush=True)

# %% CELL 4 — dung texts_title, THU TU PHAI GIONG HET E4
rows_meta, texts_title = [], []
t0 = time.time()
with open(P_CORPUS, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        cid = r["context_id"]
        if r["parse_status"] == "fallback":
            if r["dieu"]:
                d0 = r["dieu"][0]
                rows_meta.append((cid, d0.get("dieu_id", ""), d0["dieu_id"] or cid))
                texts_title.append(d0["text"])
            continue
        for dieu in r["dieu"]:
            td = (dieu.get("dieu_tieu_de") or "").strip()
            dtext = (dieu.get("text") or "").strip()
            usable = bool(td) and dtext != td
            if dieu["khoan"]:
                for k in dieu["khoan"]:
                    rows_meta.append((cid, dieu.get("dieu_id", ""), k["khoan_id"]))
                    texts_title.append(f"{td}\n{k['text']}" if usable else k["text"])
            else:
                rows_meta.append((cid, dieu.get("dieu_id", ""), dieu["dieu_id"]))
                texts_title.append(f"{td}\n{dieu['text']}" if usable else dieu["text"])
N = len(rows_meta)
print(f"corpus {N} unit trong {time.time()-t0:.0f}s", flush=True)

old_emb = np.load(P_OLD_EMB)
print("embedding CU:", old_emb.shape, old_emb.dtype, flush=True)
assert old_emb.shape[0] == N, (
    f"LECH SO HANG: corpus {N} vs embedding cu {old_emb.shape[0]}. "
    "Thu tu duyet khong khop E4 — dung lai, moi so sau day se vo nghia.")
print("  so hang khop E4 — OK", flush=True)

uid_to_row = {}
with open(P_UIDX, encoding="utf-8") as f:
    next(f)
    for line in f:
        row, cid, did, uid = line.rstrip("\n").split("\t")
        uid_to_row.setdefault(uid, int(row))
print(f"bang tra: {len(uid_to_row)} unit_id khac nhau", flush=True)

held_rows = [uid_to_row.get(r["khoan_id"], -1) for r in ho]
n_ok = sum(1 for x in held_rows if x >= 0)
print(f"giu rieng: {n_ok}/{len(ho)} cau tra duoc khoan_id ve row", flush=True)
assert n_ok >= 0.9 * len(ho), "Qua nhieu khoan_id khong tra duoc row"

# %% CELL 5 — HUAN LUYEN TRUOC, tren GPU con sach
#
# LAN 1 (20/09) CHET OOM o buoc 0/492. Nguyen nhan va ba thay doi:
#
#  a) Dung MultipleNegativesRankingLoss thuong. SAI — cong thuc Task 1 ghi ro
#     CachedMultipleNegativesRankingLoss, va ban "Cached" (GradCache) ton tai
#     DUNG DE giai bai toan bo nho nay: no chia lo thanh mini-batch, chay xuoi
#     hai lan va chi giu embedding thay vi giu ca do thi tinh toan.
#     Toi bo no vi da co negative tuong minh — hai chuyen do khong lien quan.
#  b) Do MOC truoc khi day -> embedding corpus 886 MB con nam tren GPU.
#     Nay day TRUOC, do sau.
#  c) max_seq_length 512 -> 384. Khoan trung vi 119 am tiet, p90 327;
#     384 token cat khoang 10% duoi cung thay vi 20%.
#
# So hoc bo nho tren T4 15 GB: XLM-R 568M fp32 = 2,3 GB trong so + 2,3 GB
# gradient + 4,6 GB trang thai Adam = 9,2 GB truoc khi tinh activation.
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

held_q = [r["question"] for r in ho]

model = SentenceTransformer(MODEL_NAME, device=DEVICE)
model.max_seq_length = 384

examples = []
for r in tr:
    for neg in r["negatives"]:
        examples.append(InputExample(texts=[r["question"], r["positive"], neg]))
print(f"\n{len(examples)} vi du day (moi negative mot vi du)", flush=True)

BATCH, MINI = 32, 4
loader = DataLoader(examples, shuffle=True, batch_size=BATCH, drop_last=True)
loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=MINI)
EPOCHS = 3
steps = len(loader) * EPOCHS
print(f"{EPOCHS} epoch · lo {BATCH} (mini-batch {MINI}) · {len(loader)} buoc/epoch "
      f"· tong {steps} buoc", flush=True)

t0 = time.time()
model.fit(train_objectives=[(loader, loss)],
          epochs=EPOCHS,
          warmup_steps=int(0.1 * steps),
          optimizer_params={"lr": 1e-5},
          use_amp=True,
          show_progress_bar=True)
print(f"day xong {time.time()-t0:.0f}s", flush=True)
model.save(str(OUT / "model_ft"))
torch.cuda.empty_cache()

# %% CELL 6 — do MOC (model goc, embedding E4) SAU khi da day xong
def eval_recall(corpus_emb, qemb, gold_rows, ks=(1, 5, 10, 50)):
    """Tra ve dict k -> list 0/1 tung cau, de bootstrap ghep cap.
    Chia corpus thanh lat de khong nap ca 886 MB len GPU mot luc."""
    hits = {k: [] for k in ks}
    K = max(ks)
    qt = torch.from_numpy(qemb).to(DEVICE, torch.float16)
    n = corpus_emb.shape[0]
    CH = 100_000
    best_v, best_i = [], []
    for s in range(0, n, CH):
        ce = torch.from_numpy(corpus_emb[s:s+CH]).to(DEVICE, torch.float16)
        sims = (qt @ ce.T).float()                       # [nq, lat]
        v, i = torch.topk(sims, min(K, sims.shape[1]), dim=1)
        best_v.append(v.cpu()); best_i.append((i + s).cpu())
        del ce, sims
        torch.cuda.empty_cache()
    v = torch.cat(best_v, dim=1); i = torch.cat(best_i, dim=1)
    ord_ = torch.argsort(v, dim=1, descending=True)[:, :K]
    top = torch.gather(i, 1, ord_).tolist()
    del qt
    torch.cuda.empty_cache()
    for r in range(qemb.shape[0]):
        if gold_rows[r] < 0:
            continue
        for k in ks:
            hits[k].append(1 if gold_rows[r] in top[r][:k] else 0)
    return hits

base = SentenceTransformer(MODEL_NAME, device=DEVICE)
base.max_seq_length = MAX_SEQ_LEN
base.half()
q_old = base.encode(held_q, batch_size=64, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=True).astype(np.float16)
hits_old = eval_recall(old_emb, q_old, held_rows)
print("\n=== MOC (model goc, embedding E4) ===", flush=True)
for k in (1, 5, 10, 50):
    print(f"  recall@{k:<3} {np.mean(hits_old[k]):.4f}", flush=True)
del base
torch.cuda.empty_cache()

# %% CELL 7 — nhung lai corpus bang model MOI
model.max_seq_length = MAX_SEQ_LEN
model.half()
t0 = time.time()
new_emb = model.encode(texts_title, batch_size=BATCH_SIZE_EMB,
                       normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=True).astype(np.float16)
print(f"nhung lai {new_emb.shape} trong {time.time()-t0:.0f}s", flush=True)
np.save(OUT / "corpus_embeddings_title_ft.npy", new_emb)

q_new = model.encode(held_q, batch_size=64, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=True).astype(np.float16)
hits_new = eval_recall(new_emb, q_new, held_rows)

# %% CELL 8 — SO SANH theo nguong dat truoc
def paired_bootstrap(a, b, n=2000, seed=42):
    rng = random.Random(seed)
    d = [y - x for x, y in zip(a, b)]
    k = len(d)
    m = sorted(sum(d[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return sum(d) / k, m[int(0.025 * n)], m[int(0.975 * n)]

print("\n" + "=" * 62, flush=True)
print(f"{'':10}{'goc':>10}{'fine-tune':>12}{'chenh lech':>30}")
print("-" * 62)
res = {}
for k in (1, 5, 10, 50):
    a, b = hits_old[k], hits_new[k]
    d, lo, hi = paired_bootstrap(a, b)
    res[k] = (np.mean(a), np.mean(b), d, lo, hi)
    print(f"recall@{k:<3}{np.mean(a):>10.4f}{np.mean(b):>12.4f}"
          f"{f'{d*100:+.2f} diem  CI95 [{lo*100:+.2f}, {hi*100:+.2f}]':>30}")

d50, lo50 = res[50][2], res[50][3]
c1 = d50 * 100 >= 2.0
c2 = lo50 > 0
print("\n=== NGUONG DAT TRUOC (bang nguong E4) ===")
print(f"  1. recall@50 >= +2,0 diem   {'DAT' if c1 else 'KHONG'}  ({d50*100:+.2f})")
print(f"  2. can duoi CI95 > 0        {'DAT' if c2 else 'KHONG'}  ({lo50*100:+.2f})")
print(f"\n  => {'DI TIEP: rerank lai private' if (c1 and c2) else 'DUNG: khong dem ra san xuat'}")

json.dump({str(k): {"goc": float(v[0]), "ft": float(v[1]), "delta": float(v[2]),
                    "ci_lo": float(v[3]), "ci_hi": float(v[4])} for k, v in res.items()},
          open(OUT / "p2_ketqua.json", "w"), indent=2)

# %% CELL 9 — dong goi
import shutil
shutil.make_archive("/kaggle/working/p2_model_ft", "zip", OUT / "model_ft")
print("\nfile ra:", flush=True)
for p in sorted(Path("/kaggle/working").rglob("*")):
    if p.is_file() and p.stat().st_size > 1024:
        print(f"  {p.relative_to('/kaggle/working')}  {p.stat().st_size/1e6:.1f} MB", flush=True)
