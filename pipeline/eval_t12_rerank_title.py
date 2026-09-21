"""T12 — Cham diem: reranker doc tieu de Dieu co hon reranker doc Khoan tran?

Nguon: kernel legalqa-t12-rerank-title-input. Mot lan chay, mot ro (dense-title,
dung cau hinh san xuat), hai nhanh khac nhau DUNG chu dua vao reranker:
    rr_bare   ->  u["text"]                        (san xuat hien tai)
    rr_title  ->  dieu_tieu_de + "\\n" + u["text"]
Da kiem trong kernel: rr_bare 0/50.000 cap khac ban tran, rr_title 48.892/50.000.

Mau: 1.000 cau seed 2109, tach duoc ba khoi, GIAO VOI MAU 1909 = 0. Nen day khong
phai do tren tap da dung de quet alpha/k/ngan sach.

NGUONG DAT TRUOC — ca ba phai dat:
  1. METEOR harness o cau hinh nop (alpha=0,7 k=2 ngan sach 200) >= +0,010
  2. can duoi CI95 > 0
  3. chon lai alpha tren nua A, thang tren nua B       <- bai hoc T9

Ve thu ba bat buoc: neu reranker gioi len thi alpha toi uu se DICH XUONG (tin
reranker nhieu hon). Chon alpha moi roi do tren chinh cho vua chon la lap lai T9,
noi thien lech do chon (+0,0021) lon hon chinh hieu ung (+0,0019).

Chay: PYTHONIOENCODING=utf-8 python pipeline/eval_t12_rerank_title.py
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

T12_DIR = Path("C:/Users/GIGA/AppData/Local/Temp/claude/"
               "c--Users-GIGA-VisualStudio2022Projects-job-hien-tai-dsc2026/"
               "a5d87954-e2b4-4e8f-ab41-daa8eff385fc/scratchpad/t12_out/layer3")
ARMS = ("rr_bare", "rr_title")
PROD_ALPHA = 0.7
RRF_K = 60
BUDGET = 200
K = 2
ALPHAS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_BOOTSTRAP = 2000
SEED = 42
SPLIT_SEED = 2109

MARKERS = ("Như vậy", "Theo đó", "Do đó", "Vì vậy", "Từ đó", "Như đã phân tích", "Tóm lại")
_RE = re.compile(r"(?:^|\n)\s*(" + "|".join(re.escape(m) for m in MARKERS) + r")\b")


def split_blocks(a):
    t = (a or "").replace("\r\n", "\n").strip()
    if not t or "\n" not in t:
        return "", "", ""
    nl = t.find("\n")
    lead = t[:nl].strip()
    rs = nl
    while rs < len(t) and t[rs] in "\n \t":
        rs += 1
    ms = list(_RE.finditer(t, rs))
    if not ms:
        return "", "", ""
    cut = ms[-1].start(1)
    q, c = t[rs:cut].strip(), t[cut:].strip()
    if not (lead and q and c):
        return "", "", ""
    return lead, q, c


def syl(t):
    return len(t.split())


def met(ref, hyp):
    if not hyp.strip():
        return 0.0
    return float(meteor_score([ref.split()], hyp.split(), alpha=0.9, beta=3.0, gamma=0.5))


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


def boot(diffs):
    rng = random.Random(SEED)
    n = len(diffs)
    m = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOTSTRAP))
    return statistics.fmean(diffs), m[int(0.025 * N_BOOTSTRAP)], m[int(0.975 * N_BOOTSTRAP)]


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
uid_map = {}
for _pos, (_cid, _di, _ki) in enumerate(row_map):
    _doc = parsed.get(_cid)
    if not _doc or _di >= len(_doc.get("dieu", [])):
        continue
    _dieu = _doc["dieu"][_di]
    _kl = _dieu.get("khoan", [])
    _uid = _kl[_ki]["khoan_id"] if (0 <= _ki < len(_kl)) else _dieu.get("dieu_id")
    uid_map.setdefault(_uid, _pos)
print(f"  {time.time()-t0:.0f}s", flush=True)

tcache = {}


def text_of(row):
    if row in tcache:
        return tcache[row]
    t = ""
    if 0 <= row < len(row_map):
        cid, di, ki = row_map[row]
        doc = parsed.get(cid)
        if doc and di < len(doc.get("dieu", [])):
            dieu = doc["dieu"][di]
            kl = dieu.get("khoan", [])
            if ki < 0 or not kl:
                t = dieu.get("text", "")
            else:
                t = "\n\n".join(k.get("text", "") for k in expand(kl, ki, BUDGET)
                                if k.get("text", "").strip())
    tcache[row] = t
    return t


def resolve(u):
    r = u.get("row", -1)
    if r is None or r < 0:
        r = uid_map.get(u.get("unit_id"), -1)
    return r


# --- nap hai nhanh, GIU DUNG THU TU CAU de ghep cap -------------------------
arms = {}
for arm in ARMS:
    d = {}
    with open(T12_DIR / f"L3_rerank_t12_{arm}.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            d[str(o["qid"])] = [(resolve(u), u.get("dense_rank", 1 << 20), r)
                                for r, u in enumerate(o["ranked_units"])]
    arms[arm] = d

qids = sorted(set(arms["rr_bare"]) & set(arms["rr_title"]))
cases = []
for q in qids:
    lead, quote, concl = split_blocks(train[q]["answer"])
    if quote and len(quote.split()) >= 10:
        cases.append((q, lead, quote, concl))
print(f"n = {len(cases)} câu ghép cặp được (trên {len(qids)} câu có ở cả hai nhánh)\n",
      flush=True)

# kiem bien: hai nhanh phai co thu tu KHAC nhau, nhung cung tap unit
n_same_order = 0
for q, *_ in cases:
    a = [it[0] for it in sorted(arms["rr_bare"][q], key=lambda it: it[2])]
    b = [it[0] for it in sorted(arms["rr_title"][q], key=lambda it: it[2])]
    if a == b:
        n_same_order += 1
    assert set(a) == set(b), f"[{q}] hai nhánh có tập unit KHÁC nhau — sai rổ"
print(f"KIỂM: hai nhánh cùng tập unit ở mọi câu ✓")
print(f"KIỂM: số câu reranker xếp Y HỆT nhau = {n_same_order}/{len(cases)}"
      f" ({n_same_order/len(cases):.1%})\n", flush=True)


def score(arm, alpha, subset=None):
    out = []
    for q, lead, quote, concl in cases:
        if subset is not None and q not in subset:
            continue
        items = sorted(arms[arm][q],
                       key=lambda it: -(alpha / (RRF_K + it[1])
                                        + (1 - alpha) / (RRF_K + it[2])))
        body = "\n\n".join(t for t in (text_of(it[0]) for it in items[:K]) if t.strip())
        out.append(met(f"{lead}\n{quote}\n{concl}", f"{lead}\n{body}\n{concl}"))
    return out


print("=== Quét α, cả hai nhánh ===", flush=True)
print(f"{'α':>6}" + "".join(f"{a:>12}" for a in ARMS) + f"{'chênh lệch':>14}")
print("-" * 46)
tab = {}
t0 = time.time()
for a in ALPHAS:
    for arm in ARMS:
        tab[(arm, a)] = score(arm, a)
    d = statistics.fmean(tab[("rr_title", a)]) - statistics.fmean(tab[("rr_bare", a)])
    print(f"{a:>6.1f}" + "".join(f"{statistics.fmean(tab[(arm, a)]):>12.4f}" for arm in ARMS)
          + f"{d:>+14.4f}" + ("   ← sản xuất" if a == PROD_ALPHA else ""), flush=True)
print(f"  ({time.time()-t0:.0f}s)\n")

# --- Ngưỡng 1 và 2: một biến duy nhất, cùng α sản xuất ----------------------
a0, b0 = tab[("rr_bare", PROD_ALPHA)], tab[("rr_title", PROD_ALPHA)]
d, lo, hi = boot([y - x for x, y in zip(a0, b0)])
print("=== NGƯỠNG 1 & 2 — một biến duy nhất, α = 0,7 ===")
print(f"  rr_bare   {statistics.fmean(a0):.4f}")
print(f"  rr_title  {statistics.fmean(b0):.4f}")
print(f"  chênh lệch {d:+.4f}   CI95 [{lo:+.4f}, {hi:+.4f}]\n", flush=True)

# --- Ngưỡng 3: chọn α trên nửa A, đo trên nửa B -----------------------------
idx = [q for q, *_ in cases]
random.Random(SPLIT_SEED).shuffle(idx)
half = len(idx) // 2
folds = [(set(idx[:half]), set(idx[half:])), (set(idx[half:]), set(idx[:half]))]
print("=== NGƯỠNG 3 — chọn α trên nửa A, đo trên nửa B ===")
gains = []
for fi, (fit, test) in enumerate(folds, 1):
    best_a = max(ALPHAS, key=lambda a: statistics.fmean(score("rr_title", a, fit)))
    g = statistics.fmean(score("rr_title", best_a, test)) \
        - statistics.fmean(score("rr_bare", PROD_ALPHA, test))
    gains.append(g)
    print(f"  nửa {fi}: α tốt nhất trên nửa A = {best_a}  →  lợi trên nửa B {g:+.4f}",
          flush=True)
cv = statistics.fmean(gains)
print(f"\n  LỢI THEO KIỂM CHÉO = {cv:+.4f}   (bootstrap toàn mẫu = {d:+.4f})")
print(f"  thiên lệch do chọn = {d - cv:+.4f}\n")

c1, c2, c3 = d >= 0.010, lo > 0, cv > 0
print("=== KẾT LUẬN ===")
print(f"  1. lợi ≥ +0,010             {'ĐẠT' if c1 else 'KHÔNG'}  ({d:+.4f})")
print(f"  2. cận dưới CI95 > 0        {'ĐẠT' if c2 else 'KHÔNG'}  ({lo:+.4f})")
print(f"  3. thắng trên nửa giữ riêng {'ĐẠT' if c3 else 'KHÔNG'}  ({cv:+.4f})")
print(f"\n  => {'ĐI TIẾP: rerank lại private rồi nộp' if (c1 and c2 and c3) else 'ĐÓNG — giữ 0,5383'}")
