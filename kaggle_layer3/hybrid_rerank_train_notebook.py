"""
T3b — Rerank Layer 3 tren RO HYBRID, hai nhanh, tren 1.000 cau QA_TRAIN CO GOLD.

MUC TIEU: CHAM duoc bare vs title. T3 chay tren qa_public ma tap do co 0 dap an
(da kiem: public 0/1000, private 0/1918, train 7000/7000) nen KHONG cham duoc.
T3b chay tren qa_train de co gold.

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

# Mau T3b: 1.000 cau qa_train, seed 1909, KHONG GIAO voi mau E4 (seed 2026).
# Dung lai mau E4 la do tren tap da dung de CHON title thang bare -> ket qua dep
# hon thuc te. Doi mau bien phep xac nhan thanh phep kiem doc lap. Gia: bang 0.
QIDS = ["146089", "137689", "11525", "138583", "57239", "127189", "56793", "2449", "3965", "153979", "123847", "51751", "110831", "144641", "15247", "40291", "152237", "22131", "92415", "149795", "165931", "81103", "29755", "167795", "19745", "138373", "87671", "35299", "31109", "93927", "95093", "19669", "1343", "101183", "157503", "121757", "15933", "126113", "62141", "15973", "149243", "89913", "121105", "100123", "116369", "138885", "20795", "160623", "144009", "11305", "813", "105991", "151921", "74439", "64983", "156301", "37199", "135257", "52811", "113337", "87767", "11457", "87441", "160881", "127289", "30657", "965", "34159", "40961", "147427", "57077", "154071", "109357", "14555", "95957", "73409", "125851", "65329", "40545", "67485", "147787", "66843", "154123", "130707", "123145", "45765", "103479", "99873", "139293", "5521", "71789", "126047", "156061", "79887", "57345", "79455", "159787", "129735", "46135", "97165", "17975", "86395", "34193", "139629", "100233", "15193", "101985", "116303", "50407", "13993", "45219", "69647", "120597", "122333", "107283", "103375", "164705", "19995", "20755", "36377", "34821", "151477", "32679", "39475", "143937", "120403", "50161", "78703", "26041", "22847", "135659", "34951", "124333", "3749", "10867", "5315", "9423", "50643", "160921", "36093", "68733", "87875", "82213", "77703", "168277", "64951", "141179", "107177", "84987", "39819", "31943", "21643", "156995", "111847", "86851", "73711", "30369", "28463", "121619", "23059", "155609", "139933", "42745", "27089", "17349", "48473", "12725", "273", "23353", "137011", "71613", "62647", "70731", "118539", "154451", "101845", "102857", "90275", "22081", "68279", "88797", "102521", "44197", "18093", "104795", "78085", "110657", "99733", "33849", "105275", "7603", "37633", "65969", "140829", "87993", "105797", "103601", "136979", "118351", "158501", "151021", "50739", "124901", "47037", "111437", "132225", "19753", "21503", "973", "121805", "45263", "161613", "37627", "119933", "166465", "113113", "45419", "72485", "102827", "108153", "12423", "89145", "43571", "101867", "135737", "3339", "163069", "41879", "12363", "161603", "151473", "65857", "36601", "13225", "64173", "142825", "67749", "13567", "103787", "39395", "117931", "183", "166581", "97709", "125125", "100137", "90083", "37349", "33955", "113239", "107051", "123843", "63751", "138233", "6991", "82051", "149255", "76129", "60607", "143559", "66467", "136507", "154969", "22113", "102005", "160927", "143987", "5781", "146527", "168597", "19957", "155343", "25433", "137185", "52699", "57335", "148099", "3661", "74059", "128071", "68421", "8117", "52037", "151425", "121161", "94159", "95703", "25781", "111159", "55581", "102061", "118843", "108219", "123761", "107171", "165579", "93525", "30217", "83039", "73717", "122171", "70593", "79061", "31347", "147661", "94931", "125283", "70563", "9471", "77837", "31253", "47405", "134925", "151895", "39503", "162151", "165287", "18515", "102823", "82299", "36801", "12711", "90263", "69275", "120157", "112659", "129645", "86633", "70019", "109797", "76487", "52999", "76143", "38509", "92737", "100751", "151357", "167133", "34625", "15185", "70705", "91597", "163183", "143371", "63571", "12795", "8675", "142913", "19289", "68843", "143435", "109103", "106907", "136379", "4927", "2239", "103083", "79975", "113359", "137709", "164879", "503", "69463", "120893", "134103", "85249", "143295", "54541", "679", "104673", "99003", "66661", "21659", "33691", "64089", "26929", "131255", "13673", "81819", "90631", "136515", "24109", "29985", "142797", "143889", "124057", "130739", "9519", "46657", "50281", "114919", "57273", "78387", "415", "14551", "107583", "69321", "69711", "26681", "17985", "135073", "151831", "149261", "76051", "70187", "79039", "167395", "151573", "31837", "147785", "154189", "16375", "48071", "58657", "74951", "66455", "22057", "151907", "84945", "122673", "159121", "167397", "57681", "20433", "6091", "136361", "101065", "68295", "78629", "59755", "88905", "106303", "98247", "141647", "116367", "23685", "110543", "55253", "94067", "36661", "131581", "138007", "165869", "36589", "92545", "113579", "121743", "70457", "3025", "91407", "93627", "55221", "69377", "52801", "26479", "104095", "149951", "163585", "6485", "42365", "164045", "129101", "30405", "32981", "110105", "55341", "55229", "71001", "146361", "133819", "163395", "8307", "31049", "76001", "160999", "62549", "72831", "106573", "87115", "43633", "43435", "13351", "119391", "20547", "2385", "82155", "126457", "67515", "54221", "77989", "33671", "47339", "46929", "153105", "47973", "81573", "32153", "142319", "98783", "86469", "67603", "144591", "108323", "111775", "65851", "65633", "12271", "6935", "51709", "44441", "36837", "145275", "154649", "4761", "85201", "143885", "159877", "134537", "77327", "77751", "155927", "112939", "115049", "41245", "154713", "66591", "44365", "147363", "82775", "165279", "120573", "133261", "128257", "53143", "103289", "116475", "13757", "163983", "20429", "38063", "48149", "54549", "24549", "48827", "13235", "57127", "15317", "13641", "91717", "157545", "70999", "61587", "59087", "34147", "68561", "142765", "82021", "45841", "27887", "157289", "7101", "166557", "18363", "86963", "36013", "54921", "2139", "72357", "155645", "2169", "73057", "19919", "130471", "75975", "53141", "89799", "35809", "22397", "10163", "67741", "131689", "147291", "139159", "35375", "98195", "13991", "106901", "111697", "54579", "50037", "138957", "149939", "90833", "77093", "26883", "45847", "140649", "17605", "79297", "16441", "167901", "57705", "91", "73443", "162345", "43187", "12867", "59663", "164253", "70627", "53373", "105819", "67897", "161947", "45423", "61429", "100605", "158753", "82303", "104883", "55985", "49701", "155543", "54823", "127751", "85445", "34811", "145255", "120443", "103383", "41947", "103731", "41095", "43989", "19091", "127465", "100271", "128225", "86059", "162867", "48345", "56563", "14421", "12721", "33409", "166211", "60823", "119229", "106483", "9977", "145429", "6803", "129807", "120223", "6131", "45009", "61571", "79449", "22577", "27147", "138197", "28781", "93813", "5229", "95211", "14507", "151183", "38431", "157519", "76587", "5661", "86125", "41185", "43781", "17113", "121291", "165075", "102921", "5507", "96001", "79183", "127989", "155577", "123403", "134039", "122443", "25847", "105821", "102969", "95185", "36155", "105861", "163215", "20543", "126587", "34477", "72951", "125787", "18587", "129271", "50507", "99281", "146269", "93199", "56053", "79601", "93505", "122769", "107365", "56263", "71835", "138665", "42661", "119385", "19151", "78907", "163709", "141671", "14477", "43809", "162493", "150061", "7499", "130353", "42363", "96653", "23255", "1457", "166833", "110679", "68719", "91577", "96189", "49873", "44851", "76439", "150889", "29689", "26733", "80989", "167879", "125853", "96799", "63407", "147277", "95815", "39859", "94773", "17907", "151843", "75927", "13327", "38715", "33769", "126233", "68455", "99173", "7275", "37907", "26069", "6105", "136173", "153849", "117885", "56377", "1039", "63131", "161187", "105801", "43829", "60487", "81011", "99711", "28309", "24763", "43297", "112707", "125931", "155881", "65491", "98411", "95487", "77831", "20397", "130325", "12785", "166423", "98515", "113769", "101303", "84969", "142361", "85819", "86227", "93975", "17235", "33917", "124485", "68645", "162789", "53469", "74225", "18209", "138689", "71561", "146545", "168107", "144343", "77905", "154141", "35475", "82033", "40353", "28541", "143185", "133987", "122481", "121179", "95567", "144625", "6547", "24853", "51807", "31995", "167437", "61237", "47531", "95555", "95903", "92729", "41457", "34725", "157235", "91915", "81825", "116945", "21771", "153347", "138611", "167937", "79105", "61163", "48957", "157623", "141589", "168211", "39807", "54147", "79321", "33839", "164807", "2661", "138619", "152765", "17303", "151097", "77221", "59003", "31175", "112113", "58919", "50995", "82117", "167331", "133241", "36279", "138753", "47885", "65673", "36789", "115743", "166357", "14769", "2103", "106769", "70551", "19399", "25813", "65033", "54127", "4931", "63897", "118109", "12113", "19065", "150773", "122747", "22707", "127089", "160107", "36139", "139879", "120667", "33155", "49353", "141789", "128571", "39207", "143899", "85631", "102107", "168423", "8921", "15181", "120775", "114349", "48615", "166395", "14891", "61", "147371", "152341", "37121", "144717", "27269", "122311", "13905", "81787", "53461", "69171", "138467", "29035", "159521", "72145", "6791", "20925", "2603", "157903", "41063", "46441", "49733", "165125", "40489", "125017", "39835", "157443", "133861", "116111", "21585", "140419", "125301", "9045", "26585", "135391", "23025", "84751", "131489", "160021", "96791", "130551", "18013", "75421", "135649", "11907", "161999", "166117", "78287", "29219", "118917", "63193", "124751", "109577", "150429", "130447", "37007", "127561", "167847", "84387", "106953", "9899", "33951", "39083", "67431", "161135", "140625", "67583", "10001", "22855", "77805", "148233", "99339", "96989", "14151", "148427", "155011", "149115", "58651", "9575", "90105", "67069", "163077", "137077", "108415", "91051", "139339", "165871", "141409", "153189", "75867", "64967"]


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

    out_path = OUT_DIR / f"L3_rerank_train_{arm}.jsonl"
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

shutil.make_archive("/kaggle/working/t3b_results", "zip", OUT_DIR)
print(f"\nXONG. Tong {(time.time()-t_start)/60:.1f} phut.", flush=True)
print("Tai /kaggle/working/t3b_results.zip", flush=True)
