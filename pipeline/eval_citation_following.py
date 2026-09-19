"""E5 - Doc dan chieu: keo Dieu duoc GOI TEN trong unit da lay ve vao ro.

Vi sao phai dung lai buoc CAT: neu ro = "moi Dieu cua 20 van ban BM25" thi dan chieu
trong cung van ban khong them gi. Gia tri that cua co che nay nam o cho: Dieu cua van
ban DA co trong ro nhung unit cua no ROT khoi top-50.

Khong co embedding -> dung diem trung chu cau hoi<->unit lam vat the chan cho dense.
Tuyet doi khong doc con so tuyet doi nhu recall that; chi doc CHENH LECH truoc/sau.

Do hai thuoc:
  - "voi toi duoc": co Dieu nao trong ro phu >= nguong chu cua Khoi 2 khong
  - METEOR cua thu SE GIAO DI (mo rong ngan sach 400, theo E6b)
"""
import json
import random
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
from nltk.translate.meteor_score import meteor_score

N_SAMPLE = 300
SEED = 2026
TOP_K_DOCS = 20
TOP_K_UNITS = 50
BUDGET = 400
MAX_REFS_PER_Q = 25

MARKERS = ("Nhu vay", "Theo do", "Do do", "Vi vay", "Tu do", "Nhu da phan tich", "Tom lai")
MARKERS_VN = ("Nh\u01b0 v\u1eady", "Theo \u0111\u00f3", "Do \u0111\u00f3", "V\u00ec v\u1eady",
              "T\u1eeb \u0111\u00f3", "Nh\u01b0 \u0111\u00e3 ph\u00e2n t\u00edch", "T\u00f3m l\u1ea1i")
_RE = re.compile(r"(?:^|\n)\s*(" + "|".join(re.escape(m) for m in MARKERS_VN) + r")\b")
DIEU_REF = re.compile(r"\u0110i\u1ec1u\s+(\d+[a-z]?)", re.IGNORECASE)
DOCNUM = re.compile(r"\d+/\d{4}/[A-Z\u0110][A-Z\u0110\-]*")
THIS_DOC = re.compile(r"\bn\u00e0y\b", re.IGNORECASE)


def split_blocks(a):
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


def syl(t):
    return len(t.split())


def cov(gt, text):
    if not gt:
        return 0.0
    s = set(text.lower().split())
    return sum(1 for x in gt if x in s) / len(gt)


def met(gold, hyp):
    if not hyp.strip():
        return 0.0
    return float(meteor_score([gold.split()], hyp.split(), alpha=0.9, beta=3.0, gamma=0.5))


def expand_budget(kl, idx, budget):
    if not kl:
        return []
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


def best_deliverable(dieu_list, gt):
    """Trong cac Dieu ung vien, chon phuong an giao tot nhat theo phu chu."""
    best = (0.0, "")
    for dieu in dieu_list:
        kl = dieu.get("khoan", [])
        if not kl:
            t = dieu.get("text", "")
            c = cov(gt, t)
            if c > best[0]:
                best = (c, t)
            continue
        for i in range(len(kl)):
            items = expand_budget(kl, i, BUDGET)
            t = "\n\n".join(k.get("text", "") for k in items if k.get("text", "").strip())
            c = cov(gt, t)
            if c > best[0]:
                best = (c, t)
    return best


print("Load...", flush=True)
t0 = time.time()
train = io_utils.load_train()
bm25 = retrieve.load_index(config.OUTPUTS_DIR / "bm25_doc_index.pkl")
parsed = io_utils.load_parsed_corpus()
doc_num_idx = json.loads(
    (config.OUTPUTS_DIR / "doc_number_index.json").read_text(encoding="utf-8")
)["usable"]
print(f"  {time.time()-t0:.0f}s  doc_number_index: {len(doc_num_idx)} so hieu", flush=True)

qids = [q for q in sorted(train) if train[q].get("answer")]
random.Random(SEED).shuffle(qids)

ref_stats = Counter()
rows = []
t0 = time.time()
for qid in qids:
    if len(rows) >= N_SAMPLE:
        break
    gold = split_blocks(train[qid]["answer"])
    if not gold:
        continue
    gt = gold.lower().split()
    if len(gt) < 10:
        continue
    question = train[qid]["question"]

    # --- ro goc: BM25 top-20 van ban -> moi unit -> cham diem trung chu -> cat top-50
    qset = set(question.lower().split())
    units = []
    for d in retrieve.search(bm25, question, top_k=TOP_K_DOCS):
        cid = str(d["context_id"])
        doc = parsed.get(cid)
        if not doc:
            continue
        for dieu in doc.get("dieu", []):
            kl = dieu.get("khoan", [])
            if kl:
                for k in kl:
                    kt = set(k.get("text", "").lower().split())
                    units.append((len(qset & kt) / (len(qset) + 1), cid, dieu))
            else:
                kt = set(dieu.get("text", "").lower().split())
                units.append((len(qset & kt) / (len(qset) + 1), cid, dieu))
    units.sort(key=lambda u: -u[0])
    top = units[:TOP_K_UNITS]
    base_dieu = []
    seen = set()
    for _s, cid, dieu in top:
        key = (cid, dieu.get("dieu_id"))
        if key not in seen:
            seen.add(key)
            base_dieu.append(dieu)

    # --- doc dan chieu tu text cua top-50
    extra = []
    n_ref = 0
    for _s, cid, dieu in top:
        txt = dieu.get("text", "") or ""
        for m in DIEU_REF.finditer(txt):
            if n_ref >= MAX_REFS_PER_Q:
                break
            dieu_so = m.group(1)
            win = txt[m.end():m.end() + 120]
            dn = DOCNUM.search(win)
            if dn and not THIS_DOC.search(win[:dn.start()]):
                tgt = doc_num_idx.get(dn.group(0))
                ref_stats["cross_doc"] += 1
                if tgt:
                    ref_stats["cross_doc_giai_duoc"] += 1
                else:
                    continue
            else:
                tgt = cid
                ref_stats["cung_van_ban"] += 1
            tdoc = parsed.get(str(tgt))
            if not tdoc:
                continue
            for dd in tdoc.get("dieu", []):
                if (dd.get("dieu_so") or "").strip() == dieu_so:
                    key = (str(tgt), dd.get("dieu_id"))
                    if key not in seen:
                        seen.add(key)
                        extra.append(dd)
                        n_ref += 1
                    break

    c_before, t_before = best_deliverable(base_dieu, gt)
    if extra:
        c_after, t_after = best_deliverable(base_dieu + extra, gt)
    else:
        c_after, t_after = c_before, t_before
    rows.append((c_before, c_after, met(gold, t_before), met(gold, t_after), len(extra)))
    if len(rows) % 50 == 0:
        print(f"  {len(rows)}/{N_SAMPLE} {time.time()-t0:.0f}s", flush=True)

n = len(rows)
cb = [r[0] for r in rows]
ca = [r[1] for r in rows]
mb = [r[2] for r in rows]
ma = [r[3] for r in rows]
ex = [r[4] for r in rows]

print(f"\nn={n}\n")
print("=== DAN CHIEU TIM DUOC ===")
print(f"Tham chieu trong CUNG van ban : {ref_stats['cung_van_ban']}")
print(f"Tham chieu KHAC van ban       : {ref_stats['cross_doc']} "
      f"(giai duoc so hieu: {ref_stats['cross_doc_giai_duoc']})")
print(f"Dieu them vao ro / cau: median {sorted(ex)[n//2]}  mean {statistics.fmean(ex):.1f}  "
      f"khong them gi: {sum(1 for e in ex if e==0)}/{n}")

print("\n=== VOI TOI DUOC (co Dieu phu >= nguong) ===")
for thr in (0.70, 0.90):
    nb = sum(1 for v in cb if v >= thr)
    na = sum(1 for v in ca if v >= thr)
    print(f"  phu >= {thr:.0%}:  truoc {nb:3d}/{n} = {nb/n:5.1%}   sau {na:3d}/{n} = {na/n:5.1%}"
          f"   -> {na-nb:+d} cau = {(na-nb)/n:+.1%}")

print(f"\n=== METEOR cua thu SE GIAO DI (ngan sach {BUDGET}) ===")
print(f"  truoc : {statistics.fmean(mb):.4f}")
print(f"  sau   : {statistics.fmean(ma):.4f}   chenh {statistics.fmean(ma)-statistics.fmean(mb):+.4f}")
better = sum(1 for b, a in zip(mb, ma) if a > b + 1e-9)
worse = sum(1 for b, a in zip(mb, ma) if a < b - 1e-9)
print(f"  tot hon: {better}  te hon: {worse}  khong doi: {n-better-worse}")
print(f"\nTong {time.time()-t0:.0f}s")
