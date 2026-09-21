from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path


SOURCE_BUNDLE = "__SOURCE_BUNDLE__"
SETTINGS = __RUNTIME_SETTINGS__
WORKING_ROOT = Path("/kaggle/working")
INPUT_ROOT = Path("/kaggle/input")
PRODUCTION_ROOT = WORKING_ROOT / "legalQA_Task2"
VERSION_ROOT = PRODUCTION_ROOT
OUTPUT_ROOT = WORKING_ROOT / "model_production_outputs"
HF_CACHE = Path("/root/.cache/huggingface")


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def install_dependencies() -> None:
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "transformers>=4.51,<5",
            "accelerate>=1,<2",
            "safetensors>=0.4",
            "bitsandbytes>=0.45",
            "peft>=0.17",
            "nltk>=3.9",
            "rouge-score>=0.1.2",
            "numpy>=1.26",
        ]
    )
    run([sys.executable, "-m", "pip", "uninstall", "--yes", "torchao"])
    import nltk

    nltk.download("wordnet", quiet=True, raise_on_error=True)
    nltk.download("omw-1.4", quiet=True, raise_on_error=True)


def extract_sources() -> None:
    with zipfile.ZipFile(BytesIO(base64.b64decode(SOURCE_BUNDLE))) as archive:
        for member in archive.infolist():
            destination = (WORKING_ROOT / member.filename).resolve()
            if WORKING_ROOT.resolve() not in destination.parents:
                raise RuntimeError(f"Unsafe bundled path: {member.filename}")
        archive.extractall(WORKING_ROOT)


def main() -> None:
    summary = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "settings": SETTINGS,
        "status": "running",
    }
    summary_path = WORKING_ROOT / "production_kernel_summary.json"
    try:
        os.environ["PYTHONUTF8"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["HF_HOME"] = str(HF_CACHE)
        extract_sources()
        install_dependencies()
        sys.path.insert(0, str(VERSION_ROOT / "src"))
        from legalqa_baseline.runtime_paths import resolve_input_dataset_root

        input_dataset = resolve_input_dataset_root(
            INPUT_ROOT, SETTINGS["input_dataset_slug"], SETTINGS["input_json"]
        )
        adapter = resolve_input_dataset_root(
            INPUT_ROOT, SETTINGS["adapter_slug"], "adapter_model.safetensors"
        )
        command = [
            sys.executable,
            str(PRODUCTION_ROOT / "scripts" / "run_inference.py"),
            "--config",
            str(PRODUCTION_ROOT / "configs" / "production.json"),
            "--input-json",
            str(input_dataset / SETTINGS["input_json"]),
            "--adapter-path",
            str(adapter),
            "--output-root",
            str(OUTPUT_ROOT),
            "--top-k",
            str(SETTINGS["top_k"]),
            "--seed",
            str(SETTINGS["seed"]),
            "--run-id",
            "kaggle-production",
        ]
        if SETTINGS["limit"] is not None:
            command.extend(["--limit", str(SETTINGS["limit"])])
        run(command, cwd=WORKING_ROOT)
        summary["status"] = "completed"
        summary["output_root"] = str(OUTPUT_ROOT / "kaggle-production")
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        raise
    finally:
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
