from __future__ import annotations

import argparse
import base64
import io
import json
import re
import shutil
import unicodedata
import zipfile
from pathlib import Path
from typing import Any


KAGGLE_ROOT = Path(__file__).resolve().parent
PRODUCTION_ROOT = KAGGLE_ROOT.parent
WORKSPACE_ROOT = PRODUCTION_ROOT
BASELINE_ROOT = PRODUCTION_ROOT
DEFAULT_SETTINGS = KAGGLE_ROOT / "settings.json"
DEFAULT_BUILD_ROOT = KAGGLE_ROOT / "build"


def load_settings(path: str | Path = DEFAULT_SETTINGS) -> dict[str, Any]:
    settings = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    required = {
        "schema_version",
        "username",
        "kernel_slug",
        "kernel_title",
        "accelerator",
        "model_version",
        "input_json",
        "input_dataset_slug",
        "input_dataset_title",
        "adapter_slug",
        "top_k",
        "seed",
        "limit",
    }
    missing = required - set(settings)
    unknown = set(settings) - required
    if missing or unknown:
        raise ValueError(
            f"Production Kaggle settings không hợp lệ; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    if settings["schema_version"] != 1:
        raise ValueError("Chỉ hỗ trợ schema_version=1")
    title_slug = re.sub(
        r"[^a-z0-9]+",
        "-",
        unicodedata.normalize("NFKD", settings["kernel_title"])
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower(),
    ).strip("-")
    if settings["kernel_slug"] != title_slug:
        raise ValueError(f"kernel_slug phải là {title_slug!r}")
    for slug_key in ("input_dataset_slug", "adapter_slug"):
        if settings[slug_key].split("/", 1)[0] != settings["username"]:
            raise ValueError(f"{slug_key} phải thuộc username đã cấu hình")
    if Path(settings["input_json"]).name != settings["input_json"]:
        raise ValueError("input_json trong Kaggle settings chỉ được là tên file")
    if type(settings["top_k"]) is not int or settings["top_k"] <= 0:
        raise ValueError("top_k phải là số nguyên dương")
    if type(settings["seed"]) is not int or settings["seed"] < 0:
        raise ValueError("seed phải là số nguyên không âm")
    if settings["limit"] is not None and (
        type(settings["limit"]) is not int or settings["limit"] <= 0
    ):
        raise ValueError("limit phải là null hoặc số nguyên dương")
    version_root = BASELINE_ROOT
    if not (version_root / "version.json").is_file():
        raise FileNotFoundError(f"Không tìm thấy model version: {version_root}")
    manifest = json.loads((version_root / "version.json").read_text(encoding="utf-8"))
    if manifest["model_version"] != settings["model_version"]:
        raise ValueError("model_version không khớp version.json")
    input_path = PRODUCTION_ROOT / "inputs" / settings["input_json"]
    if not input_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy production input: {input_path}")
    return settings


def _reset_target(target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_input_dataset(
    settings: dict[str, Any],
    *,
    target_root: str | Path = DEFAULT_BUILD_ROOT / "input-dataset",
) -> Path:
    target = Path(target_root).resolve()
    _reset_target(target)
    source = PRODUCTION_ROOT / "inputs" / settings["input_json"]
    shutil.copy2(source, target / settings["input_json"])
    _write_json(
        target / "dataset-metadata.json",
        {
            "id": settings["input_dataset_slug"],
            "title": settings["input_dataset_title"],
            "licenses": [{"name": "other"}],
        },
    )
    return target


def bundle_sources(settings: dict[str, Any]) -> bytes:
    version_root = BASELINE_ROOT
    files = [
        PRODUCTION_ROOT / "configs" / "production.example.json",
        PRODUCTION_ROOT / "scripts" / "run_inference.py",
        version_root / "version.json",
        version_root / "configs" / "models.json",
    ]
    files.extend(sorted((PRODUCTION_ROOT / "src").rglob("*.py")))
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Thiếu source để bundle: {missing}")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in dict.fromkeys(files):
            relative = source.relative_to(WORKSPACE_ROOT)
            if source.name == "production.example.json":
                # Runtime paths, input and adapter are supplied by the kernel CLI.
                # Keep generation options from personal config when available.
                personal = PRODUCTION_ROOT / "configs" / "production.json"
                config = json.loads((personal if personal.is_file() else source).read_text(encoding="utf-8-sig"))
                config["input_json"] = "inputs/example.json"
                config["output_root"] = "outputs"
                config["adapter"] = {"local_path": "models/adapter", "kaggle_slug": settings["adapter_slug"]}
                config["model_version"] = settings["model_version"]
                config["limit"] = None
                archive.writestr("legalQA_Task2/configs/production.json", json.dumps(config, ensure_ascii=False))
            else:
                archive.write(source, (Path("legalQA_Task2") / relative).as_posix())
    return buffer.getvalue()


def build_kernel(
    settings: dict[str, Any],
    *,
    target_root: str | Path = DEFAULT_BUILD_ROOT / "kernel",
) -> Path:
    target = Path(target_root).resolve()
    _reset_target(target)
    template = (KAGGLE_ROOT / "kernel.template.py").read_text(encoding="utf-8")
    runtime = {
        key: settings[key]
        for key in (
            "model_version",
            "input_json",
            "input_dataset_slug",
            "adapter_slug",
            "top_k",
            "seed",
            "limit",
        )
    }
    runner = template.replace(
        "__SOURCE_BUNDLE__", base64.b64encode(bundle_sources(settings)).decode("ascii")
    ).replace("__RUNTIME_SETTINGS__", repr(runtime))
    if "__SOURCE_BUNDLE__" in runner or "__RUNTIME_SETTINGS__" in runner:
        raise RuntimeError("Kernel placeholders chưa được thay thế")
    code_file = "run_legalqa_production_kaggle.py"
    (target / code_file).write_text(runner, encoding="utf-8")
    _write_json(
        target / "kernel-metadata.json",
        {
            "id": f'{settings["username"]}/{settings["kernel_slug"]}',
            "title": settings["kernel_title"],
            "code_file": code_file,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "machine_shape": settings["accelerator"],
            "dataset_sources": [
                settings["input_dataset_slug"],
                settings["adapter_slug"],
            ],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        },
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LegalQA production Kaggle assets")
    parser.add_argument(
        "command",
        choices=("validate", "build-input-dataset", "build-kernel", "build-all"),
    )
    args = parser.parse_args()
    settings = load_settings()
    if args.command == "validate":
        print(json.dumps(settings, ensure_ascii=False, indent=2))
    if args.command in {"build-input-dataset", "build-all"}:
        print(f"Built input dataset: {build_input_dataset(settings)}")
    if args.command in {"build-kernel", "build-all"}:
        print(f"Built production kernel: {build_kernel(settings)}")


if __name__ == "__main__":
    main()
