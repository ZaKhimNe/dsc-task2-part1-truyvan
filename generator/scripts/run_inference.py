from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _configure_utf8_console() -> None:
    """Keep Vietnamese CLI messages printable on Windows terminals."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


PRODUCTION_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PRODUCTION_ROOT
sys.path.insert(0, str(PRODUCTION_ROOT / "src"))

from model_production.application import execute_run  # noqa: E402


def main() -> None:
    _configure_utf8_console()
    parser = argparse.ArgumentParser(
        description="Sinh đáp án LegalQA production từ JSON retrieval của thành viên B"
    )
    parser.add_argument(
        "--config",
        default=str(PRODUCTION_ROOT / "configs" / "production.json"),
    )
    parser.add_argument("--input-json", help="Ghi đè đường dẫn input trong config")
    parser.add_argument("--output-root", help="Ghi đè thư mục output trong config")
    parser.add_argument("--adapter-path", help="Ghi đè adapter QLoRA local/Kaggle")
    parser.add_argument("--top-k", type=int, help="Số context đầu dùng cho mỗi câu")
    parser.add_argument("--limit", type=int, help="Chỉ chạy N mẫu đầu")
    parser.add_argument("--seed", type=int, help="Ghi đè seed")
    parser.add_argument("--run-id", help="Tên thư mục run; mặc định là UTC timestamp")
    parser.add_argument("--gpu-count", type=int, choices=(1, 2),
                        help="Số GPU; mặc định tự dùng tối đa 2 GPU khả dụng")
    parser.add_argument("--batch-size", type=int, default=10, help="Số câu mỗi batch trên mỗi GPU (mặc định 10)")
    args = parser.parse_args()
    run_dir = execute_run(
        args.config,
        workspace_root=WORKSPACE_ROOT,
        run_id=args.run_id,
        input_json=args.input_json,
        output_root=args.output_root,
        top_k=args.top_k,
        limit=args.limit,
        seed=args.seed,
        adapter_path=args.adapter_path,
        gpu_count=args.gpu_count,
        batch_size=args.batch_size,
    )
    print(f"Đã ghi output tại: {run_dir}")


if __name__ == "__main__":
    main()
