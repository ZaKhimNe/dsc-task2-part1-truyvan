"""E2 buoc 0 - Bug ma trung cắn bao nhieu TRONG RO THAT?

E1 do o cap corpus: 7.631 ma trung, 38.475 hang bi che (8,9%). Nhung con so do
khong noi duoc no cắn bao nhieu trong ro 50 ung vien that su duoc dung.

Cau hoi o day: trong top-50 dense that (outputs/e4/), bao nhieu suat bi mat vi
`union_units` loai trung theo `unit_id`?

  50 hang  -> bao nhieu unit_id RIENG BIET?
  Chenh lech = so suat MAT TRANG, vi hai khoan con khac nhau cung mang mot ma
  thi union chi giu duoc mot.

Neu chenh lech ~0 thi bug la chuyen ly thuyet, dong lai bang so lieu.
Neu chenh lech dang ke thi moi dang bo cong sua.

Chay: PYTHONIOENCODING=utf-8 python pipeline/measure_dup_impact_in_basket.py
"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import config

E4_DIR = config.OUTPUTS_DIR / "e4"

print("Doc unit_index.tsv...", flush=True)
row_uid = []
with open(E4_DIR / "unit_index.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        row_uid.append(parts[3])
print(f"  {len(row_uid)} hang", flush=True)

cnt = Counter(row_uid)
dup_ids = {u for u, c in cnt.items() if c > 1}
print(f"  ma bi trung: {len(dup_ids)}  |  hang mang ma trung: "
      f"{sum(cnt[u] for u in dup_ids)}", flush=True)

for arm in ("bare", "title"):
    path = E4_DIR / f"top50_{arm}_qa_train.jsonl"
    if not path.exists():
        print(f"BO QUA {arm}: khong thay {path}")
        continue
    lost = []          # so suat mat moi cau
    n_dup_rows = []    # so hang mang ma trung moi cau
    n_q = 0
    worst = (0, None)
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            rows = o["rows"]
            uids = [row_uid[r] for r in rows]
            distinct = len(set(uids))
            l = len(rows) - distinct
            lost.append(l)
            n_dup_rows.append(sum(1 for u in uids if u in dup_ids))
            if l > worst[0]:
                worst = (l, o["qid"])
            n_q += 1

    lost_sorted = sorted(lost)
    print(f"\n=== NHANH {arm.upper()} — {n_q} cau, top-50 ===")
    print(f"  Suat MAT vi trung ma: mean {statistics.fmean(lost):.2f}  "
          f"median {lost_sorted[len(lost_sorted)//2]}  "
          f"p90 {lost_sorted[9*len(lost_sorted)//10]}  max {lost_sorted[-1]}")
    print(f"  So cau mat >= 1 suat : {sum(1 for x in lost if x >= 1)}/{n_q} = "
          f"{sum(1 for x in lost if x >= 1)/n_q:.1%}")
    print(f"  So cau mat >= 5 suat : {sum(1 for x in lost if x >= 5)}/{n_q} = "
          f"{sum(1 for x in lost if x >= 5)/n_q:.1%}")
    print(f"  Hang mang ma trung   : mean {statistics.fmean(n_dup_rows):.2f}/50 = "
          f"{statistics.fmean(n_dup_rows)/50:.1%}")
    print(f"  Cau te nhat          : qid={worst[1]}, mat {worst[0]}/50 suat")

print("\n--- Doc so:")
print("  'Suat mat' = 50 - so unit_id rieng biet trong 50 hang.")
print("  Do la so ung vien bi union_units vut di vi trung ma, KHONG phai vi kem.")
