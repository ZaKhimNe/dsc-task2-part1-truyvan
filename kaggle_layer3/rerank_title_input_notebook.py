"""
T12 — RERANKER CO DOC TIEU DE DIEU KHONG? Mot bien duy nhat.

TIEN DE DA KIEM 21/09: ca ba kernel rerank (public, train, private) deu tao cap
bang `pairs.append([question, u["text"]])`. `u["text"]` la Khoan TRAN. Chuoi
"dieu_tieu_de" khong xuat hien trong file nao. Reranker CHUA BAO GIO thay tieu de.

Nhanh "title" cua T3 chi doi BO NHUNG DENSE — chinh ghi chu trong kernel do viet
"BIEN DUY NHAT: bo nhung dense". Khau rerank giu nguyen o ca hai nhanh.

GIA THUYET: day dung la benh ma E4 da chua cho dense. Khoan "Phat tien tu 200.000
den 400.000 dong" doc rieng thi giong moi Khoan phat tien trong kho; co tieu de
Dieu thi khac han. E4 gan tieu de cho dense -> hang 1 dung 56,4% -> 65,6%.

Va T3 cho thay reranker dang bi dung benh do: no co tran ~60%, keo nhanh title tu
64,8% XUONG 60,7% — ghi de thu tu tot cua dense bang y kien cua mot bo doc khong
biet Khoan thuoc chu de gi.

BIEN DUY NHAT: chu dua vao reranker.
    rr_bare   ->  u["text"]                        (hanh vi hien tai)
    rr_title  ->  dieu_tieu_de + "\n" + u["text"]
Cung MOT ro (dense-title, dung cau hinh san xuat), cung mot lan chay, cung model,
cung fp16, cung batch. Lay nhanh bare tu lan chay khac la bay H4 cua Task 1.

MAU MOI, seed 2109. KHONG dung lai mau 1909 — do la tap da dung de quet alpha, k
va ngan sach, nen do tren no la do tren cho da chinh tham so. 1.000 cau tach duoc
ba khoi (harness can the), giao voi mau cu = 0.

NGUONG DAT TRUOC — cham diem o may local, ca ba phai dat:
  1. METEOR harness o cau hinh nop (alpha=0,7 k=2 ngan sach 200) >= +0,010
  2. can duoi CI95 > 0
  3. chon lai alpha tren nua A, thang tren nua B          <- bai hoc T9
Ve thu ba bat buoc: neu reranker gioi len thi alpha toi uu se DICH XUONG (tin
reranker nhieu hon). Chon alpha moi roi do tren chinh cho vua chon la lap lai T9.

CO MAU DU: cac phep so METEOR ghep cap o n=1.000 truoc day cho CI ~ +-0,005
(vi du ngan sach 200 vs 350: +0,0070 [+0,0036; +0,0104]). Nguong +0,010 phan giai
duoc. Khac P2, noi nguong bang 9 cau ma sai so rong 12 cau.

KY VONG NOI TRUOC: +0,005 den +0,02 tren harness, nho hon tren bang xep hang.
Truot nguong thi dong, 0,5383 giu nguyen.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

t_start = time.time()

# Mau T12: 1.000 cau qa_train TACH DUOC BA KHOI, seed 2109.
# KHONG giao voi mau 1909 (da kiem: giao = 0). Mau 1909 la tap da dung de quet
# alpha, k va ngan sach — do tren no la do tren cho da chinh tham so.
QIDS = ["171", "295", "603", "735", "1069", "1189", "1235", "1489", "1623", "1747", "1795", "1949", "2087", "2175", "2687", "3287", "3313", "3541", "3601", "3605", "3639", "3703", "3801", "3887", "4179", "4275", "4295", "4409", "4475", "4521", "4791", "4819", "4857", "5217", "5433", "5483", "5809", "5857", "5919", "5975", "6239", "6307", "6395", "6609", "7227", "7243", "7317", "7477", "7517", "7731", "7753", "7995", "8075", "8237", "8331", "8379", "8381", "8485", "8593", "8743", "8919", "9033", "9253", "9581", "9599", "9651", "9745", "9763", "9849", "9969", "10095", "10319", "10641", "10893", "10945", "11339", "11441", "11565", "11613", "11771", "11881", "11937", "12899", "13173", "13229", "13233", "13247", "13319", "13341", "13371", "13495", "13611", "13627", "14573", "14619", "14763", "14831", "15325", "15361", "15959", "16001", "16265", "16661", "16681", "16769", "16795", "16887", "17133", "17155", "17885", "17903", "17937", "17953", "18027", "18039", "18071", "18481", "18745", "18915", "19129", "19169", "19267", "19605", "19905", "20021", "20379", "20463", "20619", "20881", "21165", "21513", "21549", "21797", "22193", "22299", "22371", "22757", "22761", "22827", "23383", "23797", "23931", "24015", "24171", "24205", "24253", "24481", "24621", "24701", "24745", "24859", "24897", "25001", "25121", "25139", "25143", "25245", "25563", "25651", "25877", "25963", "25991", "26265", "26269", "26277", "26673", "26949", "27117", "27151", "27521", "27525", "27591", "27627", "28195", "28401", "28457", "28477", "28479", "28519", "28577", "28863", "28919", "28981", "29117", "29171", "29203", "29317", "29881", "30363", "30773", "30981", "31023", "31203", "31385", "31521", "31523", "31581", "31749", "31789", "31839", "32215", "32247", "32535", "32573", "32685", "32709", "32743", "32923", "33041", "33059", "33065", "33193", "33411", "33507", "33949", "34351", "34411", "34475", "34783", "34973", "35063", "35129", "35343", "35511", "35765", "35959", "36191", "36283", "36953", "37077", "37315", "37495", "37509", "37563", "37613", "37755", "37801", "38029", "38087", "38259", "38309", "38901", "39039", "39075", "39115", "39315", "39519", "39673", "39743", "39901", "39935", "40231", "40927", "40983", "41251", "41335", "41513", "41589", "42135", "42163", "42211", "42249", "42655", "42919", "43205", "43661", "43851", "44113", "44295", "44801", "44867", "44877", "44925", "45083", "45155", "45727", "45769", "46111", "46191", "46507", "46653", "46655", "46761", "46917", "46931", "47265", "47393", "47577", "47919", "47967", "48109", "48225", "48571", "48633", "48823", "49131", "49947", "50571", "50683", "50701", "51209", "51555", "51593", "51595", "51731", "51827", "52409", "52457", "52593", "53169", "53495", "53689", "53697", "53885", "54173", "54175", "54267", "54399", "54597", "54783", "54803", "54841", "54845", "55083", "55311", "55547", "55595", "55841", "56611", "56617", "56621", "56697", "56709", "56837", "56921", "58113", "58263", "58433", "58753", "58825", "58981", "59005", "59161", "59399", "59495", "59541", "59585", "59701", "59811", "59881", "59933", "59959", "60161", "60263", "60347", "60527", "60569", "60735", "60737", "60811", "60847", "60931", "61241", "61269", "61373", "61543", "61583", "61669", "61793", "61861", "62077", "62267", "62533", "62649", "62757", "62915", "62937", "63133", "63145", "63283", "63431", "63467", "63633", "63743", "64035", "64311", "64317", "64719", "64727", "64799", "65749", "65779", "65809", "65833", "66391", "66613", "66899", "66927", "67245", "67265", "67407", "67465", "67721", "67751", "67759", "67915", "68021", "68695", "68859", "68875", "68975", "69085", "69137", "69141", "69915", "69991", "70167", "70185", "70623", "70923", "71559", "71997", "72007", "72327", "72439", "72813", "73343", "73405", "73541", "73679", "73691", "73813", "74041", "74525", "74639", "74729", "74775", "74915", "75037", "75171", "75223", "75317", "75383", "75525", "75555", "75965", "76203", "76283", "76299", "76431", "76563", "76759", "76963", "77107", "77159", "77175", "77409", "77603", "77679", "77749", "78217", "78279", "78699", "78807", "78829", "78839", "79041", "79371", "79723", "80029", "80081", "80511", "80631", "80685", "81027", "81043", "81727", "81755", "81845", "81879", "81905", "81993", "82185", "82539", "82541", "82611", "82623", "82633", "82899", "82917", "83017", "83411", "83649", "83929", "83939", "83957", "84253", "84575", "84825", "85103", "85465", "85579", "86007", "86105", "86335", "86365", "86379", "86519", "86757", "87061", "87229", "87571", "87581", "87643", "87663", "87809", "88019", "88157", "88961", "89069", "89109", "89233", "89395", "89545", "89695", "89717", "89743", "89977", "90159", "90261", "90291", "90499", "90911", "90931", "91187", "91245", "91357", "91557", "91785", "91809", "92511", "92531", "92697", "92725", "92743", "93071", "94283", "94451", "94573", "94751", "94827", "94855", "95037", "95155", "95261", "95275", "95359", "95389", "95557", "95695", "95735", "96491", "96595", "96611", "96819", "96985", "97161", "97289", "97427", "97453", "97539", "97669", "97741", "98227", "98449", "98459", "98499", "98667", "98785", "98819", "98831", "98895", "99223", "99469", "99947", "100045", "100197", "100237", "100263", "100827", "101151", "101163", "101583", "101889", "101933", "102167", "102213", "102313", "102347", "102363", "102409", "102927", "102977", "103335", "103501", "103923", "104181", "104225", "104251", "104491", "104525", "104603", "104713", "105101", "105207", "105359", "105487", "105565", "105677", "106165", "106235", "106709", "106723", "106933", "107615", "107617", "107627", "107703", "107975", "108073", "108161", "108283", "108511", "108663", "108689", "108843", "108899", "109307", "109319", "109503", "109621", "109943", "110155", "110261", "110503", "110621", "110627", "110645", "110841", "110933", "111057", "111177", "111629", "111713", "111827", "111901", "111931", "112063", "112145", "112223", "112261", "112351", "112467", "112473", "112555", "112769", "113125", "113217", "113367", "113371", "113825", "113857", "113909", "114123", "114163", "114225", "114251", "114915", "115059", "115089", "115133", "115213", "115235", "115263", "115365", "115565", "115577", "115757", "115893", "116079", "116371", "116383", "116545", "116553", "116569", "116769", "116873", "116947", "116951", "117345", "117453", "117511", "117637", "117671", "118005", "118213", "118613", "118811", "118853", "119435", "119469", "119569", "119575", "119909", "120629", "120657", "120765", "120845", "120963", "121067", "121135", "121261", "121335", "121491", "121539", "121601", "121887", "122125", "122267", "122353", "122373", "122621", "122653", "122721", "122907", "122961", "123127", "123461", "123949", "124035", "124201", "124263", "124315", "124619", "125193", "125205", "125259", "125531", "125907", "125915", "126265", "126377", "126379", "126389", "126395", "126483", "126663", "126881", "126885", "126989", "127137", "127315", "127339", "127371", "127675", "127711", "127727", "127739", "127913", "128069", "128149", "128279", "128447", "128811", "129159", "129535", "129571", "129725", "130215", "130303", "130529", "130541", "130585", "130705", "130717", "130877", "130915", "131053", "131095", "131117", "131207", "131479", "131555", "131587", "131807", "132267", "132757", "132819", "133075", "133087", "133201", "133221", "133389", "133497", "133653", "133785", "133843", "133965", "134339", "134363", "134501", "134521", "134531", "134545", "134605", "134613", "134745", "134953", "135335", "135535", "135651", "135733", "135855", "135981", "136057", "136261", "136367", "136847", "137063", "137159", "137193", "137211", "137615", "137737", "137791", "138189", "138235", "138465", "138989", "138991", "139059", "139101", "139217", "139415", "139429", "139589", "139703", "139883", "140113", "140199", "140471", "140693", "140719", "141085", "141459", "141795", "142645", "142709", "142869", "143099", "143163", "143253", "143417", "143563", "143655", "143887", "143995", "144095", "144179", "144759", "144969", "145129", "145395", "145471", "145535", "145649", "145713", "145731", "145759", "146029", "146341", "146399", "146645", "146703", "146921", "146957", "146965", "147111", "147477", "147511", "147869", "147977", "148227", "148337", "148351", "148479", "148525", "148731", "148909", "149315", "149617", "149853", "149997", "150103", "150797", "150877", "151047", "151131", "151617", "151681", "151951", "151969", "151985", "152617", "152807", "152971", "153233", "153641", "153671", "154369", "154511", "154633", "155041", "155279", "155367", "155539", "155673", "155977", "156155", "156195", "156371", "156411", "156443", "156551", "156875", "157029", "157245", "157433", "157597", "158313", "158965", "159083", "159323", "159843", "159893", "160331", "160815", "160865", "160907", "161255", "161285", "161459", "161495", "161595", "161619", "161679", "161851", "161987", "162175", "162333", "162393", "162553", "162593", "162607", "162617", "162733", "162897", "162959", "163049", "163117", "163185", "163197", "164055", "164057", "164117", "164245", "164495", "164521", "164643", "164787", "164853", "164983", "164997", "165073", "165165", "165285", "165625", "165701", "165793", "165897", "166063", "166177", "166273", "166353", "166439", "166647", "167507", "167663", "167917", "168001"]


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

# CHI mot bo nhung: title — dung cau hinh san xuat. Bien cua T12 nam o khau
# RERANK, khong phai khau dense.
p = list(Path("/kaggle/input").rglob("corpus_embeddings_title.npy"))
assert p, "Khong thay corpus_embeddings_title.npy (output kernel E4)"
EMB_PATH = p[0]
print(f"  nhung dense (title) <- {EMB_PATH}", flush=True)

qemb_p = list(Path("/kaggle/input").rglob("query_embeddings_qa_train.npy"))
qids_p = list(Path("/kaggle/input").rglob("query_qids_qa_train.json"))
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
qp_found = list(Path("/kaggle/input").rglob("qa_train.json"))
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
assert len(qa_public) == 7000, f"LECH: mong 7000 cau train, thay {len(qa_public)}"
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

qids_all = [q for q in QIDS if q in qa_public and q in qid_to_qemb]
print(f"  {len(qids_all)}/{len(QIDS)} cau trong mau T3b co du du lieu", flush=True)
assert len(qids_all) == 1000, f"LECH mau: mong 1000, thay {len(qids_all)}"

t0 = time.time()
corpus_emb = np.load(EMB_PATH).astype(np.float32)
print(f"  nap nhung: {corpus_emb.shape}  ({time.time()-t0:.1f}s)", flush=True)
assert corpus_emb.shape[0] == N_UNITS, "Nhung lech so hang so voi unit_index.tsv"
emb_index = retrieve.build_corpus_embedding_index(corpus_unit_ids)

basket = {}
t0 = time.time()
for i, qid in enumerate(qids_all):
    basket[qid] = retrieve.search_units_hybrid(
        bm25_index, qa_public[qid]["question"], parsed_corpus,
        corpus_emb, corpus_unit_ids, emb_index, qid_to_qemb[qid],
        top_k_docs=TOP_K_DOCS_LAYER1, dense_top_m=DENSE_TOP_M, top_k_out=TOP_K_OUT,
    )
    if i + 1 == 20:
        per_q = (time.time() - t0) / 20
        print(f"  [do thu] 20 cau {time.time()-t0:.1f}s = {per_q:.2f}s/cau"
              f"  -> ngoai suy {per_q*len(qids_all)/60:.1f} phut", flush=True)
    if (i + 1) % 250 == 0:
        print(f"    {i+1}/{len(qids_all)}  {time.time()-t0:.0f}s", flush=True)
print(f"  dung ro xong: {time.time()-t0:.1f}s", flush=True)

n_empty = sum(1 for us in basket.values() for u in us if not u["text"].strip())
sizes = [len(us) for us in basket.values()]
print(f"  KIEM: candidate text rong = {n_empty} (ky vong 0)", flush=True)
print(f"  KIEM: so ung vien/cau min={min(sizes)} max={max(sizes)} (ky vong 50)", flush=True)

# BANG TRA unit_id -> tieu de Dieu. Dung tu parsed_corpus, KHONG doan tu dinh dang
# unit_id. Ap dung dung quy tac E4: tieu de chi dung duoc khi no ton tai VA khac
# voi chinh text cua Dieu (bang nhau thi gan vao chi la lap chu).
uid_to_title = {}
n_no_title = n_eq = 0
for _cid, _d in parsed_corpus.items():
    if _d.get("parse_status") == "fallback":
        continue
    for _dieu in _d.get("dieu", []):
        _td = (_dieu.get("dieu_tieu_de") or "").strip()
        _dtext = (_dieu.get("text") or "").strip()
        _usable = bool(_td) and _dtext != _td
        if not _td:
            n_no_title += 1
        elif _dtext == _td:
            n_eq += 1
        _t = _td if _usable else ""
        if _dieu.get("khoan"):
            for _k in _dieu["khoan"]:
                uid_to_title[_k["khoan_id"]] = _t
        else:
            uid_to_title[_dieu["dieu_id"]] = _t

n_hit = sum(1 for us in basket.values() for u in us if uid_to_title.get(u["unit_id"]))
n_tot = sum(len(us) for us in basket.values())
print(f"\n  bang tra tieu de: {len(uid_to_title)} unit_id", flush=True)
print(f"  Dieu khong co tieu de        : {n_no_title}", flush=True)
print(f"  Dieu co text == dieu_tieu_de : {n_eq}", flush=True)
print(f"  ung vien THUC SU doi chu     : {n_hit}/{n_tot} = {n_hit/n_tot:.1%}", flush=True)
assert n_hit / n_tot > 0.5, (
    "Duoi 50% ung vien co tieu de dung duoc -> bien qua yeu, thi nghiem vo nghia")

# Giai phong ma tran nhung truoc khi nap reranker — T4 chi co 15 GB.
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
    for u in basket[qid]:
        trial_pairs.append([q, u["text"]])
t0 = time.time()
_ = model.predict(trial_pairs, batch_size=BATCH_SIZE, show_progress_bar=False)
dt = time.time() - t0
rate = len(trial_pairs) / dt
total_pairs = 2 * sum(len(us) for us in basket.values())   # hai nhanh RERANKER
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

# BIEN DUY NHAT cua T12: chu dua vao reranker.
#   rr_bare  -> u["text"]                           (hanh vi hien tai cua san xuat)
#   rr_title -> dieu_tieu_de + "\n" + u["text"]
# Cung ro, cung model, cung fp16, cung batch, cung MOT lan chay.
def pair_text(u, arm):
    if arm == "rr_bare":
        return u["text"]
    td = uid_to_title.get(u["unit_id"], "")
    return f"{td}\n{u['text']}" if td else u["text"]


for arm in ("rr_bare", "rr_title"):
    print(f"\n  --- rerank nhanh {arm} ---", flush=True)
    t0 = time.time()
    pairs, spans = [], []
    for qid in qids_all:
        q = qa_public[qid]["question"]
        us = basket[qid]
        spans.append((qid, len(pairs), len(us)))
        for u in us:
            pairs.append([q, pair_text(u, arm)])
    n_diff = sum(1 for (qid, st, n) in spans for j, u in enumerate(basket[qid])
                 if pairs[st + j][1] != u["text"])
    print(f"  cap co chu KHAC ban tran: {n_diff}/{len(pairs)} = {n_diff/len(pairs):.1%}",
          flush=True)
    scores = model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=True)
    dt = time.time() - t0
    print(f"  {len(pairs)} cap trong {dt:.1f}s = {len(pairs)/dt:.1f} cap/giay", flush=True)

    out_path = OUT_DIR / f"L3_rerank_t12_{arm}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for qid, start, n in spans:
            us = basket[qid]
            scored = [{
                "unit_id": u["unit_id"],
                "context_id": u["context_id"],
                "row": int(u.get("row", -1)),
                "score": float(scores[start + j]),
                "dense_rank": j,
            } for j, u in enumerate(us)]
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

shutil.make_archive("/kaggle/working/t12_results", "zip", OUT_DIR)
print(f"\nXONG. Tong {(time.time()-t_start)/60:.1f} phut.", flush=True)
print("Tai /kaggle/working/t12_results.zip", flush=True)
