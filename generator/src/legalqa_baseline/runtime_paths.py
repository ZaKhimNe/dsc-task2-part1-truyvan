from __future__ import annotations

from pathlib import Path


def resolve_input_dataset_root(
    input_root: str | Path,
    dataset_ref: str,
    required_file: str,
) -> Path:
    """Tìm thư mục dataset Kaggle mà không phụ thuộc một kiểu mount cố định."""

    root = Path(input_root)
    owner, separator, slug = dataset_ref.partition("/")
    if not separator or not owner or not slug:
        raise ValueError(f"Dataset ref không hợp lệ: {dataset_ref!r}")

    preferred_roots = (
        root / slug,
        root / "datasets" / owner / slug,
    )
    for candidate in preferred_roots:
        if (candidate / required_file).is_file():
            return candidate

    file_candidates = [
        path for path in root.rglob(required_file) if path.is_file()
    ] if root.is_dir() else []
    slug_matches = [
        path.parent
        for path in file_candidates
        if slug.casefold() in {part.casefold() for part in path.parts}
    ]
    if len(slug_matches) == 1:
        return slug_matches[0]
    if not slug_matches and len(file_candidates) == 1:
        return file_candidates[0].parent

    visible_entries = (
        sorted(path.name for path in root.iterdir()) if root.is_dir() else []
    )
    raise FileNotFoundError(
        f"Không tìm thấy {required_file!r} cho dataset {dataset_ref!r} trong "
        f"{root}; entries={visible_entries}; candidates={file_candidates}"
    )
