"""
Loader dùng chung cho QA set (train/warmup/public) và corpus (context_*.json).

Đã code sẵn (không phải khung/stub) vì I/O ở đây không cần quyết định thiết kế gì thêm —
chỉ là đọc đúng schema đã xác nhận trong docs/DATA_NOTES.md. Các module B0/B1/B2...
nên dùng loader ở đây thay vì tự mở file JSON, để mọi fallback (tên None, passage rỗng...)
chỉ xử lý ở một chỗ.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, TypedDict

from . import config


class QAItem(TypedDict):
    question: str
    answer: str | None


class ContextDoc(TypedDict):
    id: int
    name: str  # đã fallback, không bao giờ None ở đây
    name_is_fallback: bool  # True nếu tên gốc là None và phải suy từ link
    link: str
    passage: str


def load_qa_set(path: Path) -> dict[str, QAItem]:
    """Đọc train.json / warmup.json / public-official.json.

    Trả về dict {question_id(str): {"question": ..., "answer": ...}}.
    `answer` = None với public-official.json (tập cần dự đoán).
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_train() -> dict[str, QAItem]:
    return load_qa_set(config.TRAIN_PATH)


def load_warmup() -> dict[str, QAItem]:
    return load_qa_set(config.WARMUP_PATH)


def load_public_official() -> dict[str, QAItem]:
    return load_qa_set(config.PUBLIC_OFFICIAL_PATH)


def _name_from_link(link: str) -> str:
    """Fallback khi context['name'] is None — suy tên từ slug cuối URL.

    vd: '.../Nghi-dinh-16-2023-ND-CP-to-chuc-...-122042.aspx'
        -> 'Nghi-dinh-16-2023-ND-CP-to-chuc-...-122042'
    """
    slug = link.rstrip("/").split("/")[-1]
    if slug.endswith(".aspx"):
        slug = slug[: -len(".aspx")]
    return slug or "unknown"


def load_context(context_id: int | str) -> ContextDoc:
    """Đọc 1 file context_<id>.json từ data/corpus/."""
    path = config.CORPUS_DIR / f"context_{context_id}.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return _normalize_context(raw)


def _normalize_context(raw: dict) -> ContextDoc:
    name = raw.get("name")
    name_is_fallback = name is None
    if name_is_fallback:
        name = _name_from_link(raw.get("link", ""))
    return ContextDoc(
        id=raw["id"],
        name=name,
        name_is_fallback=name_is_fallback,
        link=raw.get("link", ""),
        passage=raw.get("passage", "") or "",
    )


def iter_corpus(skip_empty: bool = True) -> Iterator[ContextDoc]:
    """Duyệt toàn bộ 8,532 văn bản trong data/corpus/.

    skip_empty=True (mặc định): bỏ qua các văn bản passage rỗng
    (20 văn bản đã xác nhận trong docs/DATA_NOTES.md mục 3) — không parse/index được.
    """
    for fp in sorted(config.CORPUS_DIR.glob("context_*.json")):
        with open(fp, encoding="utf-8") as f:
            raw = json.load(f)
        doc = _normalize_context(raw)
        if skip_empty and not doc["passage"]:
            continue
        yield doc


def load_parsed_corpus() -> dict[str, dict]:
    """Đọc data/parsed_corpus.jsonl (output B1, xem src/b1_parser/parser.py)
    vào dict {context_id(str): record}. record = {context_id, name, link,
    parse_status, dieu, phu_luc_raw} — xem schema đầy đủ trong docstring B1.

    ~1.2GB RAM, ~4s đo được trên full 8,512 văn bản (12/08/2026) — load 1
    lần khi khởi động (offline), KHÔNG load lại mỗi câu hỏi."""
    result: dict[str, dict] = {}
    with open(config.DATA_DIR / "parsed_corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            result[d["context_id"]] = d
    return result


def clean_passage(text: str) -> str:
    """Lọc các cụm rác đã biết (paywall notice...). Xem config.KNOWN_NOISE_MARKERS.

    Chưa xử lý chuẩn hoá whitespace (\\r\\n lẫn \\n\\n từ crawl) — để lại cho B1 parser
    quyết định vì việc đó gắn với logic tách Điều/Khoản, không tách rời được ở tầng I/O chung.
    """
    for marker in config.KNOWN_NOISE_MARKERS:
        text = text.replace(marker, " ")
    return text
