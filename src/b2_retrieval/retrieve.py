"""
B2 — Retrieval (BM25).

Bài toán: information retrieval cổ điển — BM25 (lexical, 0 tham số học, ưu
tiên ngân sách tham số toàn hệ thống < 4.0B). Đối chiếu COLIEE Task 3
(statute retrieval): kiến trúc thắng cuộc gần đây đều BM25 filter trước,
semantic/LLM chỉ thêm sau khi có số liệu đòi hỏi.

Lưu ý phạm vi: corpus chỉ 8,532 văn bản, gần khớp tổng số câu hỏi
train+warmup+public (8,500) — nhiều khả năng là tập đã lọc sẵn, không phải
open-domain trên toàn kho luật VN (xem docs/DATA_NOTES.md mục 1, xác nhận
qua B0 khi chạy xong).

Input: question (str), tokenize bằng word-segmentation tiếng Việt
(underthesea/pyvi — KHÁC HẲN tokenizer .split() dùng để CHẤM ĐIỂM trong
common/scoring.py, hai việc độc lập, không được lẫn).

KIẾN TRÚC NHIỀU LỚP (chốt 12/08/2026, tham khảo 1 kiến trúc case-law
retrieval đã dùng trong cuộc thi khác):

  [Layer 1] BM25 cấp VĂN BẢN (context_id, không phải Khoản) — `build_index`
    + `search`/`search_docs_multi` (multi-query: full text/câu dài
    nhất/câu đầu/câu cuối). Lexical match tốt hơn ở văn bản dài.
  [Expand]  Mở rộng top-N văn bản thành list Khoản/Dieu ứng viên, đọc từ
    `parsed_corpus.jsonl` (B1) — `expand_to_units`. KHÔNG bao giờ tạo unit
    từ `phu_luc_raw`.
  [Layer 2] Bi-encoder (BGE-M3, 568M tham số) — ĐÃ ĐO (data_retrieve), CHƯA
    wire vào search_units() làm nhánh song song với Layer 1 (chỉ mới dùng
    embedding của nó cho rerank — xem dưới).
  [Layer 1.5 / rerank] `rerank_units_dense` (17/08/2026, THAY
    `rerank_units_khoan`/BM25 cũ) — chấm điểm lại candidate sau Expand bằng
    embedding Layer 2 đã tính sẵn (tra bảng + nhân, không chạy lại model).
    Đo được thắng BM25 rerank 13 điểm% Hit@1, coverage giữ nguyên — xem
    docs/EXPERIMENT_LOG.md entry [B2] 17/08. `rerank_units_khoan` GIỮ LẠI
    làm phương án dự phòng khi không có embedding sẵn.
  [Layer 3] Cross-encoder reranker (BAAI/bge-reranker-v2-m3, ~568M, cắt
    top-30 trước khi đưa vào — quá đắt nếu chạy trên toàn bộ candidate) —
    ĐANG ĐO (Kaggle GPU, cần vì reranker không tính trước được như Layer 2).

`search_units()` gộp Layer 1 + Expand, dùng ngay được cho B0 (candidate
retrieval trước khi align chính xác) — không cần chờ Layer 2/3.

Output Layer 1: list[ScoredContext] — top-K (context_id, score). Output
Expand: list[UnitCandidate] (unit_id=khoan_id/dieu_id, text, doc_score,
unit_type). Ngân sách tham số: mặc định BM25-only (0 tham số) theo B5 —
chỉ thêm Layer 2/3 nếu đo được cải thiện Recall@K đáng kể.

Workflow: build_index() 1 lần trên toàn corpus (offline) -> search_units()
mỗi khi có câu hỏi mới (online, trên đường phục vụ query).

Quyết định đã chốt (06/08/2026, xem docs/EXPERIMENT_LOG.md):
  - Thư viện BM25: `rank_bm25` (không tự viết — ưu tiên tốc độ, deadline
    submission hợp lệ đầu tiên 10/08).
  - Tokenizer tiếng Việt: `underthesea`.
  - K khởi động: 10 (tham số `top_k`, không hardcode — B3 sẽ đo lại
    5/10/20/50 bằng Recall@K khi có nhãn B0, giá trị này chỉ để chạy được
    ngay, không phải K cuối cùng).

CẢNH BÁO ĐÃ ĐO (06/08/2026, xem docs/EXPERIMENT_LOG.md entry [B2] 06/08):
`underthesea.word_tokenize` tốn RAM KHÔNG tuyến tính theo độ dài văn bản
(đo được ~1.9GB cho 1 văn bản 5.98M ký tự = `context_68843`, chính là max
trong docs/DATA_NOTES.md mục thống kê độ dài). Văn bản này là luật/quy chuẩn
thật (không phải lỗi crawl) nên KHÔNG được loại bỏ, chỉ cap độ dài đưa vào
tokenizer — xem `MAX_TOKENIZE_CHARS`. Đây là giải pháp TẠM: fix đúng về lâu
dài là B1 chunk theo Khoản trước khi index (mỗi Khoản ngắn hơn nhiều so với
cả văn bản) — B3 sẽ quyết định granularity cuối cùng.

TRẠNG THÁI: ĐÃ CODE THÂN HÀM (06/08/2026), test qua trên corpus thật (xem
docs/EXPERIMENT_LOG.md). `pip install rank_bm25 underthesea`.
"""
from __future__ import annotations

import re
from typing import Literal, Protocol, TypedDict

from rank_bm25 import BM25Okapi
from underthesea import word_tokenize

DEFAULT_TOP_K = 10
DEFAULT_TOP_K_DOCS = 20  # top-N văn bản ở Layer 1 trước khi mở rộng sang Khoản

# Cap tạm thời để tránh OOM do underthesea.word_tokenize trên văn bản cực dài
# (xem cảnh báo ở docstring module). 50k ký tự ~ p95-p97 của corpus (p90 =
# 90,141 theo DATA_NOTES.md) nên cắt rất ít văn bản, và với văn bản bị cắt,
# BM25 vẫn match tốt trên phần đầu (thường chứa tên/định nghĩa quan trọng).
MAX_TOKENIZE_CHARS = 50_000


class ScoredContext(TypedDict):
    context_id: int
    score: float


class Tokenizer(Protocol):
    """Interface tách từ tiếng Việt — cắm underthesea.word_tokenize (đã chốt)."""

    def __call__(self, text: str) -> list[str]: ...


def default_tokenizer(text: str) -> list[str]:
    """Tokenizer mặc định đã chốt: underthesea.word_tokenize."""
    return word_tokenize(text)


class BM25Index:
    """Bọc BM25Okapi + list context_id song song (BM25Okapi tự nó không giữ
    id, chỉ giữ theo vị trí — cần lớp mỏng này để search() trả lại đúng
    context_id thay vì index vị trí). Đây là 1 điều chỉnh nhỏ so với dự tính
    ban đầu ("giữ nguyên type thư viện") — phát sinh khi code thật, không
    tránh được vì rank_bm25 không có khái niệm ID tài liệu."""

    def __init__(self, bm25: BM25Okapi, context_ids: list[int]):
        self.bm25 = bm25
        self.context_ids = context_ids


def build_index(
    corpus_texts: dict[int, str],
    tokenizer: Tokenizer = default_tokenizer,
    max_tokenize_chars: int = MAX_TOKENIZE_CHARS,
) -> BM25Index:
    """Xây BM25 index (rank_bm25.BM25Okapi) từ {context_id: text_để_index}
    (text ở granularity nào do B3 quyết định — Điều nguyên khối hay Khoản).

    Văn bản dài hơn `max_tokenize_chars` bị cắt trước khi đưa vào tokenizer
    (tránh OOM do underthesea — xem cảnh báo docstring module). Đặt
    `max_tokenize_chars=None` để tắt cap (chỉ nên làm trên máy đủ RAM)."""
    context_ids = list(corpus_texts.keys())
    if max_tokenize_chars is not None:
        texts = [corpus_texts[cid][:max_tokenize_chars] for cid in context_ids]
    else:
        texts = [corpus_texts[cid] for cid in context_ids]
    tokenized_corpus = [tokenizer(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return BM25Index(bm25, context_ids)


def save_index(index: BM25Index, path) -> None:
    """Lưu BM25Index ra đĩa (pickle) — build_index trên full 8,512 văn bản
    đo được ~43 phút (12/08/2026, máy không OOM nhưng underthesea tốn chi
    phí cố định/lần gọi khá lớn) — không nên build lại mỗi lần chạy script
    khác nhau dùng đến index này."""
    import pickle

    with open(path, "wb") as f:
        pickle.dump(index, f)


def load_index(path) -> BM25Index:
    """Đọc BM25Index đã lưu bằng save_index()."""
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def search(
    index: BM25Index, query: str, tokenizer: Tokenizer = default_tokenizer, top_k: int = DEFAULT_TOP_K
) -> list[ScoredContext]:
    """Trả về top_k ScoredContext, sắp xếp giảm dần theo score."""
    tokenized_query = tokenizer(query)
    scores = index.bm25.get_scores(tokenized_query)
    ranked = sorted(zip(index.context_ids, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
    return [{"context_id": cid, "score": float(score)} for cid, score in ranked]


# ---------------------------------------------------------------------------
# Kiến trúc nhiều lớp (12/08/2026) — tham khảo 1 kiến trúc case-law retrieval
# đã dùng trong cuộc thi khác: BM25 lọc ở cấp VĂN BẢN (không phải Khoản) rồi
# mới mở rộng lấy hết Khoản/Dieu trong các văn bản đó cho tầng dense/rerank
# phía sau — lexical match tốt hơn ở văn bản dài (nhiều từ khớp hơn), thu
# hẹp về Khoản chỉ khi cần độ chính xác cao. Layer 2 (bi-encoder) + Layer 3
# (reranker, ngưỡng động) sẽ thêm sau khi có nhãn B0 thật để đo Recall@K
# từng lớp (bằng chứng cho B5) — hàm ở đây CHỈ là Layer 1 + bước Expand,
# đủ dùng ngay cho B0 làm candidate retrieval.
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+")


def build_query_variants(question: str) -> list[str]:
    """Sinh biến thể query: full text, câu dài nhất, câu đầu, câu cuối —
    câu hỏi dài có thể khớp tốt hơn với 1 câu con hơn toàn câu (ý tưởng từ
    kiến trúc case-law retrieval tham khảo). Câu hỏi chỉ 1 câu -> các biến
    thể trùng nhau, tự loại trùng bằng set."""
    question = question.strip()
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(question) if s.strip()]
    variants = {question}
    if sentences:
        variants.add(max(sentences, key=len))
        variants.add(sentences[0])
        variants.add(sentences[-1])
    return list(variants)


def search_docs_multi(
    index: BM25Index,
    question: str,
    tokenizer: Tokenizer = default_tokenizer,
    top_k_docs: int = DEFAULT_TOP_K_DOCS,
    use_multi_query: bool = False,  # ĐỔI mặc định True->False (17/08): ablation trên 7,500 câu
    # cho kết quả GIỐNG HỆT tới 4 chữ số thập phân khi bật/tắt (chỉ 1.5-3% câu hỏi có >1 mệnh
    # đề nên biến thể bị dedupe hết) — đã chốt bỏ từ 13/08 nhưng code còn để default True.
    # Giữ tham số lại (không xoá hàm) để tái đo nếu sau này gặp tập câu hỏi dài nhiều mệnh đề.
) -> list[ScoredContext]:
    """Layer 1: BM25 cấp văn bản, hợp kết quả từ nhiều biến thể query (nếu
    use_multi_query) — giữ score MAX mỗi context_id qua các biến thể, không
    cộng dồn (tránh văn bản khớp yếu ở nhiều biến thể thắng văn bản khớp rất
    mạnh ở đúng 1 biến thể)."""
    variants = build_query_variants(question) if use_multi_query else [question]
    best: dict[int, float] = {}
    for v in variants:
        for r in search(index, v, tokenizer=tokenizer, top_k=top_k_docs):
            cid = r["context_id"]
            if cid not in best or r["score"] > best[cid]:
                best[cid] = r["score"]
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k_docs]
    return [{"context_id": cid, "score": score} for cid, score in ranked]


class UnitCandidate(TypedDict):
    unit_id: str  # khoan_id hoặc dieu_id
    context_id: str
    text: str
    doc_score: float  # score BM25 Layer 1 của VĂN BẢN chứa unit này (không đổi, giữ làm tham chiếu)
    score: float  # score DÙNG ĐỂ XẾP HẠNG — = doc_score mặc định, bị ghi đè bởi rerank_units_khoan (Layer 1.5)
    unit_type: Literal["khoan", "dieu_fallback", "doc_fallback"]


def expand_to_units(doc_results: list[ScoredContext], parsed_corpus: dict[str, dict]) -> list[UnitCandidate]:
    """Bước Expand: mở rộng top-N văn bản (kết quả Layer 1) thành list
    Khoản/Dieu ứng viên, đọc từ parsed_corpus (output B1, xem
    io_utils.load_parsed_corpus()).

    unit_type:
      - "khoan": Dieu có chia Khoản -> mỗi Khoản 1 unit (trường hợp chuẩn).
      - "dieu_fallback": Dieu KHÔNG chia Khoản (vd Điều 1 câu đơn) -> cả
        Dieu.text là 1 unit.
      - "doc_fallback": văn bản parse_status="fallback" (không tách được
        Điều/Khoản) -> cả văn bản (đã trừ phu_luc_raw) là 1 unit.
    Không bao giờ tạo unit từ `phu_luc_raw` (quyết định đã chốt ở B1)."""
    units: list[UnitCandidate] = []
    for r in doc_results:
        cid = str(r["context_id"])
        doc = parsed_corpus.get(cid)
        if doc is None or not doc["dieu"]:
            continue
        if doc["parse_status"] == "fallback":
            dieu0 = doc["dieu"][0]
            units.append(
                UnitCandidate(
                    unit_id=dieu0["dieu_id"] or cid,
                    context_id=cid,
                    text=dieu0["text"],
                    doc_score=r["score"],
                    score=r["score"],
                    unit_type="doc_fallback",
                )
            )
            continue
        for dieu in doc["dieu"]:
            if dieu["khoan"]:
                for k in dieu["khoan"]:
                    units.append(
                        UnitCandidate(
                            unit_id=k["khoan_id"],
                            context_id=cid,
                            text=k["text"],
                            doc_score=r["score"],
                            score=r["score"],
                            unit_type="khoan",
                        )
                    )
            else:
                units.append(
                    UnitCandidate(
                        unit_id=dieu["dieu_id"],
                        context_id=cid,
                        text=dieu["text"],
                        doc_score=r["score"],
                        score=r["score"],
                        unit_type="dieu_fallback",
                    )
                )
    return units


def search_units(
    index: BM25Index,
    question: str,
    parsed_corpus: dict[str, dict],
    tokenizer: Tokenizer = default_tokenizer,
    top_k_docs: int = DEFAULT_TOP_K_DOCS,
    use_multi_query: bool = False,  # xem lý do đổi mặc định ở `search_docs_multi`
    khoan_index: BM25Index | None = None,
    khoan_position_index: dict[str, int] | None = None,
    corpus_embeddings=None,  # np.ndarray (N, dim), tuỳ chọn — ưu tiên dùng nếu có (xem dưới)
    corpus_embedding_index: dict[str, int] | None = None,
    query_embedding=None,  # np.ndarray (dim,), PHẢI tính sẵn bằng cùng model BGE-M3
) -> list[UnitCandidate]:
    """Layer 1 + Expand + rerank (nếu truyền đủ tham số) — trả list
    UnitCandidate (Khoản/Dieu) cho 1 câu hỏi, dùng ngay cho B0 (candidate
    retrieval trước khi align chính xác).

    2 cơ chế rerank, ƯU TIÊN DENSE nếu có đủ 3 tham số
    `corpus_embeddings`/`corpus_embedding_index`/`query_embedding` (đo được
    thắng BM25 rerank 13 điểm% Hit@1, coverage giữ nguyên — xem
    docs/EXPERIMENT_LOG.md entry [B2] 17/08). Nếu không có embedding nhưng
    có `khoan_index`/`khoan_position_index` -> dùng BM25 rerank (Layer 1.5
    bản cũ, dự phòng). Không truyền gì -> KHÔNG rerank, chỉ Layer 1 + Expand
    (tương thích ngược với caller cũ nhất).

    LƯU Ý: hàm này KHÔNG tự tính `query_embedding` (không load model BGE-M3
    ở đây, giữ b2_retrieval nhẹ, không ép caller phải cài torch nếu không
    dùng dense) — caller tự tính sẵn (vd tra bảng embedding câu hỏi đã có
    sẵn cho tập câu hỏi cố định, hoặc gọi model riêng nếu câu hỏi mới)."""
    doc_results = search_docs_multi(index, question, tokenizer, top_k_docs, use_multi_query)
    units = expand_to_units(doc_results, parsed_corpus)
    if corpus_embeddings is not None and corpus_embedding_index is not None and query_embedding is not None:
        units = rerank_units_dense(corpus_embeddings, corpus_embedding_index, query_embedding, units)
    elif khoan_index is not None and khoan_position_index is not None:
        units = rerank_units_khoan(khoan_index, khoan_position_index, question, units, tokenizer)
    return units


# ---------------------------------------------------------------------------
# Layer 1.5 (13/08/2026) — rerank rẻ bằng BM25 cấp Khoản, KHÔNG search lại
# toàn bộ 393,999 unit (đo được 2.68s/câu, quá chậm — xem
# docs/EXPERIMENT_LOG.md entry [B2] 13/08). Thay vào đó dùng
# `BM25Okapi.get_batch_scores()` (rank_bm25) — chấm điểm CHỈ trên tập
# candidate đã có (vài nghìn unit từ Expand), nhưng vẫn dùng thống kê
# IDF/avgdl của TOÀN CORPUS Khoản (394k unit, đã build sẵn trong
# outputs/bm25_khoan_index.pkl) — đúng ý nghĩa BM25 hơn tự build index nhỏ
# tạm thời từ riêng tập candidate (IDF sẽ lệch, không đại diện toàn corpus).
#
# Ablation trên data_retrieve (13/08) đo được: BM25 cấp Khoản sắc nét hơn
# hẳn cấp văn bản khi xếp hạng (Hit@1 +5.8 điểm%, MRR +24%) nhưng search
# thẳng cấp Khoản mất 7% coverage (từ khóa rải rác nhiều Khoản trong cùng
# văn bản, tách nhỏ mất hiệu ứng cộng dồn). Layer 1.5 lấy được cái sắc nét
# đó MÀ KHÔNG mất coverage, vì Layer 1 (cấp văn bản) đã đảm bảo candidate
# đầu vào — chỉ rerank thứ tự trong tập đó, không tự tìm thêm.
# ---------------------------------------------------------------------------


def build_khoan_position_index(khoan_index: BM25Index) -> dict[str, int]:
    """{unit_id: vị trí trong khoan_index.context_ids} — cần để gọi
    get_batch_scores(doc_ids=[vị trí,...]), API của rank_bm25 dùng vị trí
    nguyên (int), không dùng thẳng unit_id string. Build 1 lần, dùng lại
    cho mọi câu hỏi (không phải việc rẻ nếu build lại mỗi lần — 394k entry)."""
    return {uid: i for i, uid in enumerate(khoan_index.context_ids)}


def rerank_units_khoan(
    khoan_index: BM25Index,
    khoan_position_index: dict[str, int],
    question: str,
    units: list[UnitCandidate],
    tokenizer: Tokenizer = default_tokenizer,
) -> list[UnitCandidate]:
    """Layer 1.5: chấm điểm lại `units` (candidate từ Expand) bằng BM25 cấp
    Khoản, ghi đè field `score` (giữ nguyên `doc_score` làm tham chiếu Layer
    1 cũ). Trả về list ĐÃ SẮP XẾP GIẢM DẦN theo score mới.

    Unit không có trong khoan_index (hiếm — lệch id, hoặc `unit_type=
    "doc_fallback"`/"dieu_fallback" của văn bản build sau khi build
    bm25_khoan_index) -> giữ nguyên score cũ (doc_score), coi như không
    rerank được case đó, KHÔNG loại bỏ (an toàn hơn mất candidate)."""
    positions: list[int] = []
    idx_by_position: list[int] = []  # vị trí trong `units` tương ứng từng phần tử `positions`
    for i, u in enumerate(units):
        pos = khoan_position_index.get(u["unit_id"])
        if pos is not None:
            positions.append(pos)
            idx_by_position.append(i)

    if not positions:
        return sorted(units, key=lambda u: u["score"], reverse=True)

    tokenized_query = tokenizer(question)
    new_scores = khoan_index.bm25.get_batch_scores(tokenized_query, positions)

    reranked = list(units)
    for i, score in zip(idx_by_position, new_scores):
        reranked[i] = {**reranked[i], "score": float(score)}
    reranked.sort(key=lambda u: u["score"], reverse=True)
    return reranked


# ---------------------------------------------------------------------------
# Dense rerank (17/08/2026) — THAY `rerank_units_khoan` (BM25) làm cơ chế
# rerank chính, dùng embedding Khoản đã tính sẵn cho Layer 2
# (outputs/layer2/corpus_embeddings.npy, 432,473 unit) thay vì BM25.
#
# Người dùng đặt đúng câu hỏi: BM25 "khá yếu" để rerank, sao không dùng
# thẳng embedding đã có sẵn từ Layer 2 — kiểm chứng trên data thật (không
# đoán): test trên CÙNG 1 tập candidate (Layer 1 BM25-doc top-100 -> Expand,
# giữ nguyên, chỉ đổi cách CHẤM ĐIỂM LẠI), warmup.json 500 câu:
#   BM25-Khoản (rerank_units_khoan cũ): Hit@1=32.2% MRR=0.452
#   Dense (hàm này, tra bảng embedding có sẵn): Hit@1=45.2% MRR=0.564
# Coverage GIỐNG HỆT (89.8% cả 2 — đúng lý thuyết, rerank không đổi TẬP
# candidate, chỉ đổi THỨ TỰ) nhưng dense hơn 13 điểm% ở Hit@1 — thắng rõ
# ràng, MIỄN PHÍ (embedding đã tính sẵn 1 lần cho Layer 2, ở đây chỉ tra vị
# trí + nhân 1 phép, không cần chạy lại model BGE-M3) — xem
# docs/EXPERIMENT_LOG.md entry [B2] 17/08. `rerank_units_khoan`/BM25 GIỮ
# LẠI (không xoá) làm phương án dự phòng khi không có embedding sẵn.
# ---------------------------------------------------------------------------


def build_corpus_embedding_index(corpus_unit_ids: list[str]) -> dict[str, int]:
    """{unit_id: vị trí (hàng) trong ma trận embedding} — cần để tra
    `corpus_embeddings[vị trí]` đúng unit, giống ý nghĩa
    `build_khoan_position_index` nhưng cho embedding thay vì BM25Index."""
    return {uid: i for i, uid in enumerate(corpus_unit_ids)}


def rerank_units_dense(
    corpus_embeddings,  # np.ndarray (N, dim), đã chuẩn hoá (normalize_embeddings=True lúc tạo)
    corpus_embedding_index: dict[str, int],
    query_embedding,  # np.ndarray (dim,), CÙNG model + đã chuẩn hoá như corpus_embeddings
    units: list[UnitCandidate],
) -> list[UnitCandidate]:
    """Chấm điểm lại `units` bằng dense embedding đã tính sẵn (Layer 2),
    ghi đè field `score` (giữ nguyên `doc_score` làm tham chiếu). Trả về
    list ĐÃ SẮP XẾP GIẢM DẦN theo score mới (cosine similarity, vì cả 2 phía
    đã chuẩn hoá nên chỉ cần dot product).

    `query_embedding` phải tính SẴN bằng cùng model (BGE-M3) đã dùng tạo
    `corpus_embeddings` — hàm này KHÔNG tự chạy model, chỉ tra bảng + nhân,
    rẻ hơn nhiều so với gọi lại model cho từng unit.

    Unit không có trong `corpus_embedding_index` (hiếm) -> giữ nguyên score
    cũ, không loại bỏ (an toàn hơn mất candidate — nhất quán với
    `rerank_units_khoan`)."""
    import numpy as np

    positions: list[int] = []
    idx_by_position: list[int] = []
    for i, u in enumerate(units):
        pos = corpus_embedding_index.get(u["unit_id"])
        if pos is not None:
            positions.append(pos)
            idx_by_position.append(i)

    if not positions:
        return sorted(units, key=lambda u: u["score"], reverse=True)

    sub_embeddings = corpus_embeddings[positions]
    new_scores = sub_embeddings @ np.asarray(query_embedding)

    reranked = list(units)
    for i, score in zip(idx_by_position, new_scores):
        reranked[i] = {**reranked[i], "score": float(score)}
    reranked.sort(key=lambda u: u["score"], reverse=True)
    return reranked


# ---------------------------------------------------------------------------
# HYBRID Union cấp UNIT + RRF fusion (17/08/2026) — sửa 3 lỗi kiến trúc tìm
# ra khi rà soát (xem docs/EXPERIMENT_LOG.md entry [B2] 17/08 "RÀ SOÁT"):
#
#  Lỗi 2 — Union cũ định làm ở cấp VĂN BẢN (thiết kế bị nhiễm từ cách ĐO:
#    nhãn data_retrieve chỉ có cấp văn bản). Hệ quả: dense đã biết chính xác
#    Khoản nào tốt nhưng ta bỏ đi, thu về doc_id, rồi expand lại TOÀN BỘ
#    Khoản của doc đó -> rác chen chỗ. `union_units()` gộp thẳng ở CẤP UNIT.
#
#  Lỗi 3 — xếp hạng chỉ bằng 1 thang điểm (dense) thì vùi lấp candidate mà
#    nhánh kia đóng góp riêng (với 51 câu BM25 tìm được/dense trượt, dense
#    score là thước đo TỆ). `fuse_units_rrf()` dùng Reciprocal Rank Fusion —
#    chuẩn ngành cho việc gộp 2 bảng xếp hạng có thang điểm KHÔNG so sánh
#    được, 0 tham số học.
#
#  Lỗi 1 — điểm cắt trước Layer 3 chọn tuỳ ý (30) làm trần recall tụt xuống
#    76.0%. `cut_units_diverse()` thêm ràng buộc "tối đa N Khoản/văn bản" —
#    đo được +5.6 điểm% MIỄN PHÍ ở cùng 30 slot (76.0% -> 81.6%), vì đáp án
#    nằm ở 1 VĂN BẢN cụ thể nên tiêu nhiều slot cho nhiều Khoản cùng văn bản
#    là dư thừa. CẢNH BÁO: số đo đó ở cấp văn bản — max_per_doc=1 có thể giữ
#    đúng văn bản nhưng SAI Khoản, cần nhãn B0 cấp Khoản để chốt (chưa làm).
# ---------------------------------------------------------------------------

RRF_K = 60  # hằng số chuẩn của RRF (Cormack et al.) — làm mượt ảnh hưởng của hạng đầu


def union_units(*unit_lists: list[UnitCandidate]) -> list[UnitCandidate]:
    """Gộp nhiều list UnitCandidate ở CẤP UNIT (không collapse về văn bản),
    loại trùng theo `unit_id` — giữ bản GẶP ĐẦU TIÊN (list truyền vào trước
    được ưu tiên giữ metadata như doc_score).

    Thứ tự trả về không mang ý nghĩa xếp hạng — phải chấm điểm/fuse sau
    (xem `fuse_units_rrf`)."""
    merged: dict[str, UnitCandidate] = {}
    for units in unit_lists:
        for u in units:
            if u["unit_id"] not in merged:
                merged[u["unit_id"]] = u
    return list(merged.values())


def fuse_units_rrf(
    ranked_lists: list[list[UnitCandidate]],
    rrf_k: int = RRF_K,
    weights: list[float] | None = None,
) -> list[UnitCandidate]:
    """Reciprocal Rank Fusion: gộp nhiều BẢNG XẾP HẠNG (mỗi list đã sort
    giảm dần theo thang điểm RIÊNG của nó) thành 1 xếp hạng duy nhất.

    score_rrf(u) = Σ_i weight_i / (rrf_k + rank_i(u))   (rank 1-indexed)

    Unit không xuất hiện trong 1 bảng nào đó -> bảng đó đóng góp 0 (hành vi
    chuẩn của RRF). Dùng RANK thay vì SCORE nên KHÔNG cần chuẩn hoá thang
    điểm giữa BM25 (vài chục) và cosine dense (0-1) — đúng lý do chọn RRF.

    Trả về list unit đã sort giảm dần theo score_rrf, field `score` ghi đè
    bằng score_rrf (giữ `doc_score` làm tham chiếu Layer 1)."""
    if not ranked_lists:
        return []
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    rrf_scores: dict[str, float] = {}
    unit_by_id: dict[str, UnitCandidate] = {}
    for w, units in zip(weights, ranked_lists):
        for rank, u in enumerate(units, start=1):
            uid = u["unit_id"]
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + w / (rrf_k + rank)
            if uid not in unit_by_id:
                unit_by_id[uid] = u

    fused = [{**unit_by_id[uid], "score": score} for uid, score in rrf_scores.items()]
    fused.sort(key=lambda u: u["score"], reverse=True)
    return fused


def search_units_hybrid(
    bm25_index: BM25Index,
    question: str,
    parsed_corpus: dict[str, dict],
    corpus_embeddings,
    corpus_unit_ids: list[str],
    corpus_embedding_index: dict[str, int],
    query_embedding,
    top_k_docs: int = 100,
    dense_top_m: int = 300,
    top_k_out: int | None = 50,
    tokenizer: Tokenizer = default_tokenizer,
) -> list[UnitCandidate]:
    """THIẾT KẾ CUỐI của B2 (chốt 17/08/2026, mọi lựa chọn có số đo hậu thuẫn
    — xem docs/EXPERIMENT_LOG.md entry [B2/B4] 17/08):

        Layer 1 (BM25 doc top_k_docs) → Expand ─┐
                                                  ├─ Union CẤP UNIT
        Layer 2 (dense unit-level top dense_top_m)┘
          → xếp hạng TOÀN BỘ bằng dense score
          → cắt top_k_out (KHÔNG ràng buộc đa dạng)

    Vì sao từng lựa chọn (đo trên nhãn B0 cấp Khoản, n=393):
      • Union 2 nhánh: BM25 và dense trượt ở NHỮNG CÂU KHÁC NHAU (chỉ 2.6%
        cùng trượt) — bổ trợ thật.
      • Xếp hạng bằng dense (KHÔNG phải RRF): Hit Khoản@3 = 55.5% vs RRF
        27.0%. RRF chỉ hơn +2.8 điểm% ở pool K=50, không bù nổi -28.5 ở K=3.
      • KHÔNG ràng buộc đa dạng: `max_per_doc=1` giữ đúng văn bản (98.7%)
        nhưng SAI Khoản — Hit Khoản sụp 81.9% → 43.3%.
      • top_k_out=50: pool cấp Khoản 85.0%, là tập đưa vào Layer 3.
        `top_k_out=None` -> trả hết (dùng khi caller tự cắt).

    Trả về list UnitCandidate đã sort giảm dần theo `score` (= dense score).
    """
    doc_results = search_docs_multi(bm25_index, question, tokenizer, top_k_docs, use_multi_query=False)
    units_lexical = expand_to_units(doc_results, parsed_corpus)
    units_dense = dense_search_units(
        corpus_embeddings, corpus_unit_ids, query_embedding, dense_top_m
    )
    merged = union_units(units_lexical, units_dense)
    ranked = rerank_units_dense(corpus_embeddings, corpus_embedding_index, query_embedding, merged)
    out = ranked if top_k_out is None else ranked[:top_k_out]
    # Unit CHỈ có ở nhánh dense không mang text (xem dense_search_units) — phải điền
    # trước khi trả ra, nếu không B4/B6 sẽ giao text RỖNG cho Generator (bug thật đã
    # bắt được khi test 17/08). Chỉ điền cho các unit THỰC SỰ trả ra (top_k_out), không
    # phải cả tập union — tránh tra cứu vô ích.
    return [u if u["text"] else {**u, "text": get_unit_text(parsed_corpus, u["unit_id"])} for u in out]


def get_unit_text(parsed_corpus: dict[str, dict], unit_id: str) -> str:
    """Tra text của 1 unit theo `unit_id` từ `parsed_corpus` (B1 output).

    Format unit_id (xem b1_parser): `{context_id}_{dieu_so}_{khoan_so}` (Khoản),
    `{context_id}_{dieu_so}` (Điều không chia Khoản), `{context_id}_` (fallback
    toàn văn). Trả "" nếu không tìm thấy (không raise — an toàn hơn cho pipeline).

    LƯU Ý: 7,632 `khoan_id` bị TRÙNG trong corpus (danh sách đánh số lồng nhau
    trong cùng 1 Điều — xem docs/EXPERIMENT_LOG.md entry [B1] 13/08) → hàm này
    trả bản KHỚP ĐẦU TIÊN. Chấp nhận được vì các bản trùng thuộc cùng 1 Điều
    của cùng 1 văn bản, nội dung liên quan nhau."""
    context_id = unit_id.split("_", 1)[0]
    doc = parsed_corpus.get(context_id)
    if doc is None:
        return ""
    for dieu in doc["dieu"]:
        if dieu["dieu_id"] == unit_id:
            return dieu["text"]
        for k in dieu["khoan"]:
            if k["khoan_id"] == unit_id:
                return k["text"]
    # fallback: unit_id kiểu "{context_id}_" của văn bản parse_status="fallback"
    if doc["dieu"]:
        return doc["dieu"][0]["text"]
    return ""


def dense_search_units(
    corpus_embeddings, corpus_unit_ids: list[str], query_embedding, top_m: int
) -> list[UnitCandidate]:
    """Layer 2: dense tìm ĐỘC LẬP trên TOÀN corpus (không qua BM25), trả về
    top_m unit — giữ nguyên là UNIT, KHÔNG collapse về văn bản (xem Lỗi 2
    trong docs/EXPERIMENT_LOG.md entry [B2] 17/08 "RÀ SOÁT").

    `text` để rỗng — unit từ nhánh này chỉ cần `unit_id` để union/xếp hạng;
    caller tự tra text từ `parsed_corpus` khi cần (tránh giữ 432k text trong
    RAM chỉ để lấy vài trăm)."""
    import numpy as np

    scores = corpus_embeddings @ np.asarray(query_embedding)
    top_m = min(top_m, len(corpus_unit_ids) - 1)
    top_idx = np.argpartition(-scores, top_m)[:top_m]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    units: list[UnitCandidate] = []
    for i in top_idx:
        uid = corpus_unit_ids[i]
        units.append(
            UnitCandidate(
                unit_id=uid,
                context_id=uid.split("_", 1)[0],
                text="",
                doc_score=float(scores[i]),
                score=float(scores[i]),
                unit_type="khoan",
            )
        )
    return units


def cut_units_diverse(
    units: list[UnitCandidate], top_k: int, max_per_doc: int | None = None
) -> list[UnitCandidate]:
    """Cắt còn `top_k` unit, mỗi văn bản góp tối đa `max_per_doc` unit (giữ
    thứ tự xếp hạng đầu vào). `max_per_doc=None` (MẶC ĐỊNH) -> không giới hạn.

    ⚠️ `max_per_doc=1` NGHE HẤP DẪN NHƯNG CÓ HẠI — đã đo (17/08/2026):
      • Ở cấp VĂN BẢN: top_k=30 + max_per_doc=1 cho Hit 81.6% vs 76.0% khi
        không giới hạn (+5.6 điểm%, phủ 30 văn bản thay vì 9.9) → trông như
        cải thiện miễn phí.
      • Ở cấp KHOẢN (mới là cái pipeline cần — nhãn B0, n=393): Hit Khoản
        SỤP **81.9% → 43.3%**. Giữ đúng văn bản (98.7%) nhưng SAI Khoản.
    Nguyên nhân: mỗi văn bản chỉ được giữ 1 Khoản — thường không phải Khoản
    chứa đáp án. Bài học: đo ở đúng cấp mà pipeline thực sự cần.
    Xem docs/EXPERIMENT_LOG.md entry [B2/B4] 17/08."""
    if max_per_doc is None:
        return units[:top_k]
    out: list[UnitCandidate] = []
    per_doc: dict[str, int] = {}
    for u in units:
        cid = u["context_id"]
        if per_doc.get(cid, 0) >= max_per_doc:
            continue
        per_doc[cid] = per_doc.get(cid, 0) + 1
        out.append(u)
        if len(out) >= top_k:
            break
    return out
