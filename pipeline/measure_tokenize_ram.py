"""Do RAM dinh khi tokenize o CAC MUC CAP khac nhau — TRUOC khi build lai index.

VI SAO: comment retrieve.py:60-66 ghi RAM ton KHONG TUYEN TINH theo do dai, do duoc
~1,9 GB cho mot van ban 5,98 trieu ky tu. Khong tuyen tinh nghia la KHONG suy ra duoc
200.000 ky tu ton bao nhieu. Co the re hon nhieu, co the co nguong gay o dau do.

Do mot phat o dung cau hinh sap chay, con hon build 43 phut roi chet OOM o phut 38.
Day la quy tac 6 cua Task 1 — ho dinh hai lan vi bo qua no.

Chay: PYTHONIOENCODING=utf-8 python pipeline/measure_tokenize_ram.py
"""
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import io_utils

try:
    import psutil
    PROC = psutil.Process()

    def rss_mb():
        return PROC.memory_info().rss / 1024 / 1024
except ImportError:
    PROC = None

    def rss_mb():
        return float("nan")

CAPS = [50_000, 100_000, 200_000, 400_000]

print("Tim van ban dai nhat...", flush=True)
t0 = time.time()
longest = ("", 0)
lens = []
for d in io_utils.iter_corpus():
    n = len(d["passage"])
    lens.append(n)
    if n > longest[1]:
        longest = (d["passage"], n)
text, L = longest
lens.sort()
print(f"  {len(lens)} van ban, {time.time()-t0:.0f}s")
print(f"  dai nhat: {L:,} ky tu")
print(f"  median {lens[len(lens)//2]:,}  p90 {lens[9*len(lens)//10]:,}  "
      f"p99 {lens[99*len(lens)//100]:,}\n", flush=True)

if PROC is None:
    print("!! khong co psutil -> chi do THOI GIAN, khong do duoc RAM")
    print("   cai: pip install psutil\n", flush=True)

from src.b2_retrieval.retrieve import default_tokenizer  # noqa: E402

print(f"{'cap (ky tu)':>14}{'thoi gian':>12}{'RAM dinh them':>16}{'so token':>12}")
print("-" * 56)
base = rss_mb()
for cap in CAPS:
    if cap > L:
        print(f"{cap:>14,}  (vuot do dai van ban dai nhat, bo qua)")
        continue
    gc.collect()
    before = rss_mb()
    t0 = time.time()
    try:
        toks = default_tokenizer(text[:cap])
        dt = time.time() - t0
        peak = rss_mb()
        print(f"{cap:>14,}{dt:>11.1f}s{peak-before:>15.0f}MB{len(toks):>12,}", flush=True)
        del toks
    except MemoryError:
        print(f"{cap:>14,}   MemoryError — nguong gay o day", flush=True)
        break
    except Exception as e:
        print(f"{cap:>14,}   LOI {type(e).__name__}: {str(e)[:50]}", flush=True)
        break
    gc.collect()

print(f"\nRSS goc truoc khi do: {base:.0f}MB")
print("\n--- DOC SO ---")
print("So can nhin la RAM DINH THEM o muc 200.000, va thoi gian mot van ban.")
print("Nhan thoi gian voi so van ban vuot cap (918) de uoc chi phi them cua build lai.")
print("Neu 200.000 van re thi nang cap; neu gay thi lui ve 100.000.")
print("\nCACH DUNG VE LAU DAI (chinh comment de xuat): chunk theo Khoan TRUOC khi")
print("index, moi Khoan ngan hon nhieu so voi ca van ban. Khong lam giua tuan private.")
