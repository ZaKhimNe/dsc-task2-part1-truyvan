"""P2 buoc 2 — Them HARD NEGATIVE cung Dieu / cung van ban.

VI SAO BUOC NAY BAT BUOC, va vi sao cong thuc Task 1 khong du:

Task 1 dung CachedMultipleNegativesRankingLoss voi negative TRONG LO - tuc cac
van ban ngau nhien cua cau hoi khac. O muc VAN BAN thi the la du: hai van ban
ngau nhien khac han nhau.

O muc KHOAN thi khong du. Viec ta can model hoc la TACH CAC KHOAN TRONG CUNG MOT
DIEU ra khoi nhau - chung cung chu de, cung tu vung phap ly, chi khac chi tiet.
Negative ngau nhien khong bao gio day duoc dieu do: model chi can hoc "cau hoi ve
thue thi gan Khoan ve thue" la da thang het negative trong lo.

Nen: moi positive duoc gan 1-3 negative la Khoan KHAC trong CUNG Dieu (uu tien),
hoac cung van ban (du phong). Do phu voi Khoi 2 gold phai THAP de chac chan no
khong phai dap an bi bo sot.

Chay: PYTHONIOENCODING=utf-8 python pipeline/build_khoan_hard_negatives.py
"""
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import config, io_utils

SEED = 2026
N_NEG = 3
MAX_NEG_COVERAGE = 0.35      # negative phai phu THAP -> chac chan khong phai dap an
LEN_MATCH = True             # chon negative gan do dai positive nhat (xem ghi chu trong vong lap)
MIN_NEG_TOKENS = 20
IN = config.OUTPUTS_DIR / "khoan_pairs_train.jsonl"
OUT = config.OUTPUTS_DIR / "khoan_triplets_train.jsonl"


def toks(t):
    return t.lower().split()


def coverage(gold_tokens, cand_text):
    if not gold_tokens:
        return 0.0
    cand = set(toks(cand_text))
    return sum(1 for t in gold_tokens if t in cand) / len(gold_tokens)


print("Load...", flush=True)
parsed = io_utils.load_parsed_corpus()
train = io_utils.load_train()
pairs = [json.loads(l) for l in open(IN, encoding="utf-8")]
print(f"  {len(pairs)} cặp gốc\n", flush=True)

rng = random.Random(SEED)
stats = Counter()
rows = []

for p in pairs:
    doc = parsed.get(p["context_id"])
    if not doc:
        stats["không tìm thấy văn bản"] += 1
        continue
    gold_tokens = toks(p["khoan_text"])

    pos_len = max(len(p["khoan_text"].split()), 1)

    same_dieu, same_doc = [], []
    for dieu in doc.get("dieu", []):
        is_same = dieu.get("dieu_id") == p["dieu_id"]
        for k in dieu.get("khoan", []):
            if k.get("khoan_id") == p["khoan_id"]:
                continue
            t = k.get("text", "")
            if len(t.split()) < MIN_NEG_TOKENS:
                continue
            if coverage(gold_tokens, t) > MAX_NEG_COVERAGE:
                continue
            (same_dieu if is_same else same_doc).append(t)

    # CAN BANG DO DAI - bat buoc.
    # Do kiem 20/09 tren ban dau tien: positive dai gap 6 lan negative, va mot bo
    # phan loai CHI DUNG DO DAI dat 94,3%. Model se hoc "dai hon = dung" thay vi
    # hoc ngu nghia, roi hong hoan toan o san xuat noi 432k Khoan cung canh tranh.
    # Nen chon negative CO DO DAI GAN POSITIVE NHAT, khong chon ngau nhien.
    def closest(pool, n):
        return sorted(pool, key=lambda t: abs(math.log(max(len(t.split()), 1) / pos_len)))[:n]

    negs = closest(same_dieu, N_NEG)
    if len(negs) < N_NEG:
        negs += closest(same_doc, N_NEG - len(negs))

    if not negs:
        stats["không có negative nào"] += 1
        continue

    stats[f"{len(negs)} negative"] += 1
    stats["có ít nhất 1 negative cùng Điều"] += bool(same_dieu)
    rows.append({
        "qid": p["qid"],
        "question": p["question"],
        "positive": p["khoan_text"],
        "negatives": negs,
        "khoan_id": p["khoan_id"],
        "dieu_id": p["dieu_id"],
        "f1": p["f1"],
        "n_same_dieu": min(len(same_dieu), N_NEG),
    })

with open(OUT, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("=== KẾT QUẢ ===")
for k, v in sorted(stats.items()):
    print(f"  {k:<38} {v:>6}")
print(f"\n  ĐÃ GHI {len(rows)} triplet -> {OUT}")

if rows:
    nd = sum(r["n_same_dieu"] for r in rows)
    tot = sum(len(r["negatives"]) for r in rows)
    print(f"\n  tổng negative            {tot}")
    print(f"  trong đó CÙNG ĐIỀU       {nd}  ({nd/tot:.1%})  <- phần dạy model tách Khoản")
    print(f"  câu có ≥1 negative cùng Điều: {stats['có ít nhất 1 negative cùng Điều']}"
          f"/{len(rows)}  ({stats['có ít nhất 1 negative cùng Điều']/len(rows):.1%})")
    nl = sorted(len(n.split()) for r in rows for n in r["negatives"])
    pl = sorted(len(r["positive"].split()) for r in rows)
    print(f"\n  độ dài positive (âm tiết): trung vị {pl[len(pl)//2]}")
    print(f"  độ dài negative (âm tiết): trung vị {nl[len(nl)//2]}")

    print(f"\n=== TÁCH TẬP: giữ riêng để đo, KHÔNG dạy ===")
    idx = list(range(len(rows)))
    random.Random(SEED).shuffle(idx)
    n_hold = len(rows) // 5
    hold = set(idx[:n_hold])
    for name, sel in (("train", [i for i in idx if i not in hold]),
                      ("heldout", [i for i in idx if i in hold])):
        path = config.OUTPUTS_DIR / f"khoan_triplets_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i in sel:
                f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
        print(f"  {name:<8} {len(sel):>5} triplet -> {path.name}")
    print("\n  Mọi con số báo cáo về P2 phải đo trên heldout. Đo trên train là đo trí nhớ.")

    # PHEP KIEM BAT BUOC. Neu bo phan loai chi dung DO DAI ma van doan dung cao
    # thi du lieu day mo hinh mot loi tat, khong phai ngu nghia. Ban dau tien
    # (nhan recall, negative khong can bang) dat 94,3% -> phai vut.
    win = tie = lose = 0
    for r in rows:
        a = len(r["positive"].split())
        for n in r["negatives"]:
            b = len(n.split())
            if a > b:
                win += 1
            elif a == b:
                tie += 1
            else:
                lose += 1
    tot2 = win + tie + lose
    acc = max(win, lose) / tot2
    print("\n=== PHÉP KIỂM BẮT BUỘC: bộ phân loại CHỈ DÙNG ĐỘ DÀI ===")
    print(f"  positive dài hơn negative : {win}/{tot2} = {win/tot2:.1%}")
    print(f"  ngắn hơn                  : {lose}/{tot2} = {lose/tot2:.1%}")
    # MOC SO SANH DUNG khong phai 50%.
    # Do 20/09 tren 18.177 cap (gold vs MOI Khoan khac cung Dieu, khong chon loc):
    # gold dai hon 74,2% mot cach TU NHIEN - cau hoi nham vao Khoan noi dung chinh,
    # va Khoan noi dung chinh vốn dai hon cac Khoan dinh nghia/dan chieu.
    # Nen muc tieu KHONG phai keo ve 50% (lam vay la bop meo du lieu), ma la
    # khong VUOT muc tu nhien. Ban dau 94,3% >> 74,2% -> that su hong.
    NATURAL = 0.742
    print(f"\n  đoán bằng độ dài đạt {acc:.1%}")
    print(f"  mức TỰ NHIÊN trong kho (18.177 cặp cùng Điều) = {NATURAL:.1%}")
    print("  bản đầu tiên (nhãn recall, negative không cân bằng) = 94,3%  <- vượt xa tự nhiên")
    print(f"\n  => {'DÙNG ĐƯỢC — không vượt mức tự nhiên' if acc <= NATURAL else 'HỎNG — vượt mức tự nhiên'}")
