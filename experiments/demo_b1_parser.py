"""
Demo B1 — parse_document trên data thật. Chạy: python experiments/demo_b1_parser.py

Test 1: context_740 (Quyết định 5868/QĐ-BYT, mẫu chuẩn) -> in ra toàn bộ cây
  Điều/Khoản, kiểm tra bằng mắt tiêu đề không bị cắt cụt.
Test 2: mẫu ngẫu nhiên 30 văn bản (seed=42) -> đo tỉ lệ match cấu trúc vs fallback,
  assert không văn bản nào có title chứa "\\n" lạ (dấu hiệu bug cắt cụt đã gặp).
Test 3: case có Phụ lục (context_166280, context_231867) -> assert không còn
  Khoản giả (số bảng biểu bị bắt nhầm) dính trong Điều cuối.
Test 4: edge case rỗng/không cấu trúc.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config, io_utils
from src.b1_parser import parser


def test_context_740():
    print("=== Test 1: context_740 (Quyết định 5868/QĐ-BYT) ===")
    d = io_utils.load_context(740)
    parsed = parser.parse_document("740", d["passage"])
    dieu_list = parsed["dieu"]
    assert parsed["parse_status"] == "matched"
    assert len(dieu_list) == 5, f"Kỳ vọng 5 Điều, được {len(dieu_list)}"
    assert dieu_list[2]["dieu_tieu_de"] == "Cơ cấu tổ chức và hoạt động", (
        f"Title Điều 3 sai (bug cắt cụt?): {dieu_list[2]['dieu_tieu_de']!r}"
    )
    for dieu in dieu_list:
        assert "\n" not in dieu["dieu_tieu_de"], f"Title dính \\n lạ: {dieu!r}"
        assert dieu["dieu_id"] == f"740_{dieu['dieu_so']}"
        print(f"  {dieu['dieu_id']}: {dieu['dieu_tieu_de']!r} ({len(dieu['khoan'])} khoản)")
    print("  PASS\n")


def test_sample_30():
    print("=== Test 2: mẫu ngẫu nhiên 30 văn bản (seed=42) ===")
    random.seed(42)
    all_ids = [int(p.stem.split("_")[1]) for p in config.CORPUS_DIR.glob("context_*.json")]
    sample = random.sample(all_ids, 30)

    n_ok, n_fallback, n_empty = 0, 0, 0
    for cid in sample:
        p = io_utils.load_context(cid)["passage"]
        parsed = parser.parse_document(str(cid), p)
        dieu_list = parsed["dieu"]
        if not p.strip():
            assert dieu_list == [], f"{cid}: passage rỗng nhưng không trả []"
            n_empty += 1
            continue
        is_fallback = parsed["parse_status"] == "fallback"
        n_fallback += is_fallback
        n_ok += not is_fallback
        for dieu in dieu_list:
            assert "\n" not in dieu["dieu_tieu_de"], f"{cid}: title dính \\n lạ: {dieu['dieu_tieu_de']!r}"

    print(f"  {len(sample)} văn bản: khớp cấu trúc={n_ok}, fallback={n_fallback}, rỗng={n_empty}")
    print("  PASS (không có title bị cắt cụt/dính \\n lạ)\n")


def test_phu_luc_cases():
    print("=== Test 3: case có Phụ lục (context_166280, context_231867) ===")
    for cid in (166280, 231867):
        p = io_utils.load_context(cid)["passage"]
        parsed = parser.parse_document(str(cid), p)
        assert parsed["phu_luc_raw"], f"{cid}: kỳ vọng có phu_luc_raw"
        for dieu in parsed["dieu"]:
            for k in dieu["khoan"]:
                assert len(k["text"]) >= 3, (
                    f"{cid} {dieu['dieu_id']}: Khoản giả lọt qua: {k!r}"
                )
        print(f"  context_{cid}: phu_luc_raw tách riêng ({len(parsed['phu_luc_raw'])} ký tự), "
              f"không còn Khoản giả")
    print("  PASS\n")


def test_edge_cases():
    print("=== Test 4: edge case ===")
    assert parser.parse_document("0", "")["dieu"] == []
    assert parser.parse_document("0", "   \n\n  ")["dieu"] == []
    fallback = parser.parse_document("0", "Đoạn văn thường, không Điều Khoản gì.")
    assert fallback["parse_status"] == "fallback"
    assert len(fallback["dieu"]) == 1 and fallback["dieu"][0]["dieu_so"] == ""
    print("  PASS\n")


if __name__ == "__main__":
    test_context_740()
    test_sample_30()
    test_phu_luc_cases()
    test_edge_cases()
    print("Tất cả test PASS.")
