"""E6b — quet ngan sach lai, voi Khoan xuat phat chon KHONG thien vi do dai.

Van de cua E6: chon Khoan bang phu chu (recall) -> Khoan DAI luon phu nhieu hon
-> median Khoan xuat phat 396 am tiet trong khi median that cua corpus la 49.
Hau qua: phan lon cau da vuot ngan sach san, expand() khong chay.

Ba bo chon, tinh trong CUNG mot luot:
  A recall    |cand ∩ gold| / |gold|     (cu, thien vi Khoan DAI)
  B f1        hai chieu                   (trung hoa do dai)
  C precision |cand ∩ gold| / |cand|      (= "text nam GON trong Khoi 2",
                                            dung dinh nghia nhan theo chuoi)
"""
import random, re, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import config, io_utils
from src.b2_retrieval import retrieve
from nltk.translate.meteor_score import meteor_score

N_SAMPLE, SEED, TOP_K_DOCS, LEN_CAP = 300, 2026, 20, 1200
BUDGETS = [0, 50, 100, 150, 200, 250, 300, 400, 500, 700, 1000]
MIN_KHOAN_TOKENS = 5   # chan Khoan sieu ngan an diem precision

MARKERS = ("Như vậy","Theo đó","Do đó","Vì vậy","Từ đó","Như đã phân tích","Tóm lại")
_RE = re.compile(r"(?:^|\n)\s*(" + "|".join(re.escape(m) for m in MARKERS) + r")\b")

def split_blocks(a):
    t=(a or "").replace("\r\n","\n").strip()
    if not t: return ""
    nl=t.find("\n")
    if nl==-1: return ""
    rs=nl
    while rs<len(t) and t[rs] in "\n \t": rs+=1
    ms=list(_RE.finditer(t,rs))
    if not ms: return ""
    q=t[rs:ms[-1].start(1)].strip()
    return q if (t[:nl].strip() and q) else ""

def syl(t): return len(t.split())

def expand(kl, idx, budget):
    if budget<=0: return [kl[idx]]
    lo=hi=idx; total=syl(kl[idx].get("text",""))
    while total<budget:
        ch,cl = hi+1<len(kl), lo-1>=0
        if not ch and not cl: break
        if ch: hi+=1; total+=syl(kl[hi].get("text",""))
        else:  lo-=1; total+=syl(kl[lo].get("text",""))
    return kl[lo:hi+1]

def met(gold,hyp):
    if not hyp.strip(): return 0.0
    return float(meteor_score([gold.split()], hyp.split(), alpha=0.9, beta=3.0, gamma=0.5))

def cov(gt,text):
    return sum(1 for t in gt if t in set(text.lower().split()))/len(gt) if gt else 0.0

print("Load...",flush=True); t0=time.time()
train=io_utils.load_train(); bm25=retrieve.load_index(config.OUTPUTS_DIR/"bm25_doc_index.pkl")
parsed=io_utils.load_parsed_corpus(); print(f"  {time.time()-t0:.0f}s",flush=True)

qids=[q for q in sorted(train) if train[q].get("answer")]
random.Random(SEED).shuffle(qids)

sel={"A recall":[], "B f1":[], "C precision":[]}
t0=time.time(); n_done=0
for qid in qids:
    if n_done>=N_SAMPLE: break
    gold=split_blocks(train[qid]["answer"])
    if not gold: continue
    gt=gold.lower().split()
    if len(gt)<10: continue
    G=set(gt)
    best={k:(-1.0,None,None) for k in sel}
    for d in retrieve.search(bm25, train[qid]["question"], top_k=TOP_K_DOCS):
        doc=parsed.get(str(d["context_id"]))
        if not doc: continue
        for dieu in doc.get("dieu",[]):
            kl=dieu.get("khoan",[])
            for i,k in enumerate(kl):
                kt=k.get("text","").lower().split()
                if len(kt)<MIN_KHOAN_TOKENS: continue
                K=set(kt); inter=len(K&G)
                if not inter: continue
                r=inter/len(G); p=inter/len(K); f=2*p*r/(p+r)
                for name,v in (("A recall",r),("B f1",f),("C precision",p)):
                    if v>best[name][0]: best[name]=(v,dieu,i)
    if best["B f1"][1] is None: continue
    for name in sel:
        if best[name][1] is not None: sel[name].append((gold,best[name][1],best[name][2]))
    n_done+=1
    if n_done%50==0: print(f"  {n_done}/{N_SAMPLE} {time.time()-t0:.0f}s",flush=True)

print(f"\nn={n_done}\n",flush=True)

for name in ("A recall","B f1","C precision"):
    cases=sel[name]
    start_len=sorted(syl(c[1]["khoan"][c[2]].get("text","")) for c in cases)
    print(f"=== BO CHON {name}  (n={len(cases)}, Khoan xuat phat median "
          f"{start_len[len(start_len)//2]} am tiet) ===")
    print(f"{'ngan sach':>10} {'METEOR':>8} {'phu chu':>8} {'dai median':>11} {'>cap':>6}")
    res=[]
    for B in BUDGETS:
        ms,cs,ls=[],[],[]
        for gold,dieu,idx in cases:
            items=expand(dieu.get("khoan",[]),idx,B)
            txt="\n\n".join(k.get("text","") for k in items if k.get("text","").strip())
            ms.append(met(gold,txt)); cs.append(cov(gold.lower().split(),txt)); ls.append(syl(txt))
        ls.sort(); m=statistics.fmean(ms); res.append((B,m))
        print(f"{B:>10} {m:>8.4f} {statistics.fmean(cs):>8.3f} {ls[len(ls)//2]:>11} "
              f"{sum(1 for l in ls if l>LEN_CAP)/len(ls):>5.1%}")
    ms=[met(g,d.get("text","")) for g,d,_ in cases]
    ls=sorted(syl(d.get("text","")) for _,d,_ in cases)
    print(f"{'ca Dieu':>10} {statistics.fmean(ms):>8.4f} {'':>8} {ls[len(ls)//2]:>11} "
          f"{sum(1 for l in ls if l>LEN_CAP)/len(ls):>5.1%}")
    b=max(res,key=lambda r:r[1])
    print(f"  -> DINH: {b[0]} am tiet, METEOR {b[1]:.4f} (Khoan tran {res[0][1]:.4f}, "
          f"chenh {b[1]-res[0][1]:+.4f})\n",flush=True)
print("V8 dang dat PER_ITEM_BUDGET_SYLLABLES = 350")
