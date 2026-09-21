from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Callable


def validate_packages(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("Input phải là một danh sách JSON không rỗng")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Mẫu thứ {index} phải là object")
        sample_id = str(row.get("id", "")).strip()
        question = row.get("question")
        contexts = row.get("contexts")
        if not sample_id:
            raise ValueError(f"Mẫu thứ {index} thiếu id")
        if sample_id in seen:
            raise ValueError(f"ID trùng: {sample_id}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Mẫu {sample_id} thiếu question")
        if not isinstance(contexts, list) or not contexts:
            raise ValueError(f"Mẫu {sample_id} thiếu contexts")
        for context_index, context in enumerate(contexts):
            if not isinstance(context, dict):
                raise ValueError(
                    f"Context {context_index} của mẫu {sample_id} phải là object"
                )
            if not isinstance(context.get("text"), str) or not context["text"].strip():
                raise ValueError(
                    f"Context {context_index} của mẫu {sample_id} thiếu text"
                )
        seen.add(sample_id)
        validated.append(row)
    return validated


def _append_unique(target: list[str], value: Any) -> None:
    if not isinstance(value, str):
        return
    normalized = " ".join(value.split())
    if normalized and normalized.casefold() not in {
        item.casefold() for item in target
    }:
        target.append(normalized)


def _document_name(context: dict[str, Any]) -> str | None:
    document = context.get("document")
    if isinstance(document, str):
        return document
    if isinstance(document, dict):
        for key in ("name", "title", "document_name"):
            value = document.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def build_context_bundle(row: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k phải là số nguyên dương")
    contexts = row.get("contexts")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError(f"Mẫu {row.get('id', '<unknown>')} thiếu contexts")
    selected = contexts[:top_k]
    metadata = {
        "document_numbers": [],
        "named_documents": [],
        "articles": [],
        "clauses": [],
        "points": [],
    }
    texts: list[str] = []
    seen_texts: set[str] = set()
    for context in selected:
        if not isinstance(context, dict):
            raise ValueError("Mỗi context phải là object")
        text = context.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Mỗi context phải có text không rỗng")
        stripped = text.strip()
        normalized = " ".join(stripped.split()).casefold()
        # Context trùng nhau (vd 2 Khoản cùng 1 Điều sau khi mở rộng) không
        # được cộng thêm vào Khối 2 — Khối 2 được chép nguyên văn vào câu trả
        # lời cuối nên trùng lặp ở đây trực tiếp làm câu trả lời lặp ý, tốn
        # ngân sách token mà không tăng recall thật.
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        texts.append(stripped)
        _append_unique(metadata["document_numbers"], context.get("document_number"))
        _append_unique(metadata["named_documents"], _document_name(context))
        _append_unique(metadata["articles"], context.get("article"))
        _append_unique(metadata["clauses"], context.get("clause"))
        _append_unique(metadata["points"], context.get("point"))
    return {
        "text": "\n\n".join(texts),
        "citation_metadata": metadata,
        "selected_context_count": len(texts),
        "available_context_count": len(contexts),
    }


def build_submission(results: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    submission: dict[str, dict[str, str]] = {}
    for result in results:
        sample_id = str(result.get("id", "")).strip()
        answer = result.get("answer")
        if not sample_id or not isinstance(answer, str):
            raise ValueError("Mỗi kết quả phải có id và answer dạng chuỗi")
        if sample_id in submission:
            raise ValueError(f"ID trùng trong kết quả: {sample_id}")
        submission[sample_id] = {"answer": answer}
    return submission


def score_if_available(
    submission: dict[str, dict[str, str]],
    rows: list[dict[str, Any]],
    *,
    scorer: Callable[[dict, dict], dict[str, Any]],
) -> dict[str, Any]:
    references = {
        str(row["id"]): row.get("reference_answer")
        for row in rows
        if isinstance(row.get("reference_answer"), str)
        and row["reference_answer"].strip()
    }
    if not references:
        return {
            "scoring_status": "not_available_no_reference",
            "meteor": None,
            "rouge_l": None,
            "scored_count": 0,
        }
    if len(references) != len(rows):
        return {
            "scoring_status": "not_available_incomplete_reference",
            "meteor": None,
            "rouge_l": None,
            "scored_count": 0,
        }
    scores = scorer(submission, references)
    return {
        "scoring_status": "completed",
        "meteor": float(scores["meteor"]),
        "rouge_l": float(scores["rouge"]),
        "scored_count": int(scores.get("count", len(rows))),
    }


def resolve_model_version(workspace_root: str | Path, model_version: str) -> Path:
    workspace = Path(workspace_root).resolve()
    if not model_version or Path(model_version).name != model_version:
        raise ValueError(f"model_version không hợp lệ: {model_version!r}")
    version_root = workspace
    required = [
        version_root / "src" / "legalqa_baseline",
        version_root / "configs" / "models.json",
        version_root / "version.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Phiên bản model {model_version!r} chưa đầy đủ: {missing}"
        )
    manifest = json.loads((version_root / "version.json").read_text(encoding="utf-8"))
    if manifest["model_version"] != model_version:
        raise FileNotFoundError(f"Model version {model_version!r} không có trong project")
    return version_root
