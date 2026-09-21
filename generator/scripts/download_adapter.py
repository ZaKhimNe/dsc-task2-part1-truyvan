"""Download the configured Kaggle adapter dataset into a local model directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "production.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    adapter = config["adapter"]
    target = (ROOT / adapter["local_path"]).resolve()
    if target.exists():
        raise FileExistsError(f"Adapter target already exists; choose a new local_path: {target}")
    with tempfile.TemporaryDirectory(prefix="legalqa-adapter-") as temporary:
        subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download", "-d",
             adapter["kaggle_slug"], "-p", temporary, "--unzip"], check=True,
        )
        candidates = [p.parent for p in Path(temporary).rglob("adapter_config.json")
                      if (p.parent / "adapter_model.safetensors").is_file()]
        if len(candidates) != 1:
            raise ValueError(f"Expected one adapter, found {len(candidates)}")
        manifest = json.loads((candidates[0] / "adapter_config.json").read_text(encoding="utf-8"))
        models = json.loads((ROOT / "configs" / "models.json").read_text(encoding="utf-8"))
        expected = models[config["model_key"]]["model_id"]
        if manifest.get("base_model_name_or_path") != expected:
            raise ValueError(f"Adapter base model does not match {expected}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(candidates[0], target)
    print(f"Adapter ready: {target}")


if __name__ == "__main__":
    main()
