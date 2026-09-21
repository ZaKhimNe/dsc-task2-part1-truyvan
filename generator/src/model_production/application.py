from __future__ import annotations

import hashlib
import importlib
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .pipeline import (
    build_context_bundle,
    build_submission,
    resolve_model_version,
    score_if_available,
    validate_packages,
)


REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "model_version",
    "model_key",
    "input_json",
    "output_root",
    "top_k",
    "seed",
    "limit",
    "prompt_mode",
    "decoding_mode",
    "enable_quality_retry",
    "adapter",
}


def _resolve_path(production_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (production_root / path).resolve()


def load_config(
    config_path: str | Path,
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Production config phải là object JSON")
    missing = REQUIRED_CONFIG_KEYS - set(config)
    if missing:
        raise ValueError(f"Production config thiếu khóa: {sorted(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("Chỉ hỗ trợ production config schema_version=1")
    if type(config["top_k"]) is not int or config["top_k"] <= 0:
        raise ValueError("top_k phải là số nguyên dương")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed phải là số nguyên không âm")
    if config["limit"] is not None and (
        type(config["limit"]) is not int or config["limit"] <= 0
    ):
        raise ValueError("limit phải là null hoặc số nguyên dương")
    adapter = config["adapter"]
    if not isinstance(adapter, dict) or not {
        "local_path",
        "kaggle_slug",
    }.issubset(adapter):
        raise ValueError("adapter phải có local_path và kaggle_slug")
    resolve_model_version(workspace_root, str(config["model_version"]))
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mean_flag(details: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(bool(item["generation"].get(key)) for item in details)


def _select_gpu_count(requested: int | None, available: int) -> int:
    if requested is not None and (type(requested) is not int or requested not in (1, 2)):
        raise ValueError("gpu_count phải là 1 hoặc 2")
    if requested == 2 and available < 2:
        raise ValueError(f"Yêu cầu 2 GPU nhưng chỉ có {available} GPU khả dụng")
    return min(available, requested or 2)


def _generate_shard(indexed_rows, generator_options, top_k, seed, *,
                    device_id=None, generator_factory=None, batch_size=10):
    # Each spawned process owns a model and a CUDA device; RNG state is isolated.
    if device_id is not None:
        import torch
        torch.cuda.set_device(device_id)
    if generator_factory is None:
        from legalqa_baseline.generation import QwenGenerator
        generator_factory = QwenGenerator
    from legalqa_baseline.blocks import assemble_answer
    generator = generator_factory(**generator_options)
    details = []
    for start in range(0, len(indexed_rows), batch_size):
        chunk = indexed_rows[start:start + batch_size]
        contexts = [build_context_bundle(row, top_k=top_k) for _, row in chunk]
        items = [dict(question=row["question"], context=context["text"],
                      citation_metadata=context["citation_metadata"],
                      seed=seed + index, fewshot_examples=[])
                 for (index, row), context in zip(chunk, contexts)]
        if batch_size > 1 and hasattr(generator, "generate_batch"):
            generations = generator.generate_batch(items)
        else:
            generations = [generator.generate(**item) for item in items]
        if len(generations) != len(chunk):
            raise RuntimeError("Batch trả về sai số lượng câu trả lời")
        for (index, row), context, generated in zip(chunk, contexts, generations):
            details.append((index, {
                "id": str(row["id"]), "question": row["question"], "context": context,
                "generation": generated,
                "answer": assemble_answer(generated["lead"], context["text"], generated["conclusion"]),
                "has_reference": bool(row.get("reference_answer")),
            }))
            print(f"[device={device_id if device_id is not None else 'cpu'}] "
                  f"index={index} id={row['id']} valid={generated.get('format_valid')} "
                  f"retry={generated.get('retry_used')}", flush=True)
    return details, {
        "adapter_path": str(getattr(generator, "adapter_path", generator_options.get("adapter_path"))),
        "parameter_count": getattr(generator, "parameter_count", None),
    }


def execute_run(
    config_path: str | Path,
    *,
    workspace_root: str | Path,
    generator_factory: Callable[..., Any] | None = None,
    run_id: str | None = None,
    input_json: str | Path | None = None,
    output_root: str | Path | None = None,
    top_k: int | None = None,
    limit: int | None = None,
    seed: int | None = None,
    adapter_path: str | Path | None = None,
    gpu_count: int | None = None,
    batch_size: int = 10,
) -> Path:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size phải là số nguyên dương")
    workspace = Path(workspace_root).resolve()
    production_root = workspace
    config = load_config(config_path, workspace_root=workspace)
    version_root = resolve_model_version(workspace, str(config["model_version"]))
    source_root = version_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    scoring = importlib.import_module("legalqa_baseline.scoring")

    selected_input = Path(input_json).resolve() if input_json else _resolve_path(
        production_root, str(config["input_json"])
    )
    selected_output = Path(output_root).resolve() if output_root else _resolve_path(
        production_root, str(config["output_root"])
    )
    selected_top_k = top_k if top_k is not None else int(config["top_k"])
    selected_limit = limit if limit is not None else config["limit"]
    selected_seed = seed if seed is not None else int(config["seed"])
    selected_adapter = (
        Path(adapter_path).resolve()
        if adapter_path
        else _resolve_path(production_root, str(config["adapter"]["local_path"]))
    )
    if not selected_input.is_file():
        raise FileNotFoundError(f"Không tìm thấy input JSON: {selected_input}")

    rows = validate_packages(
        json.loads(selected_input.read_text(encoding="utf-8-sig"))
    )
    if selected_limit is not None:
        rows = rows[: int(selected_limit)]
    if not rows:
        raise ValueError("Không có mẫu để sinh đáp án")

    models = json.loads(
        (version_root / "configs" / "models.json").read_text(encoding="utf-8")
    )
    model_key = str(config["model_key"])
    if model_key not in models:
        raise KeyError(f"model_key không tồn tại trong phiên bản: {model_key}")
    model_config = models[model_key]
    if generator_factory is None and not selected_adapter.is_dir():
        raise FileNotFoundError(f"Không tìm thấy adapter QLoRA: {selected_adapter}")
    available_gpus = 0
    if generator_factory is None or gpu_count is not None:
        import torch
        available_gpus = torch.cuda.device_count()
    selected_gpus = min(_select_gpu_count(gpu_count, available_gpus), len(rows))
    worker_count = max(1, selected_gpus)
    generator_options = dict(
        config=model_config, cache_dir=version_root / ".cache" / "huggingface",
        adapter_path=selected_adapter, prompt_mode=str(config["prompt_mode"]),
        decoding_mode=str(config["decoding_mode"]),
        enable_quality_retry=bool(config["enable_quality_retry"]),
    )
    started = datetime.now(timezone.utc)
    identifier = run_id or started.strftime("%Y%m%dT%H%M%SZ")
    run_dir = selected_output / identifier
    if run_dir.exists():
        raise FileExistsError(f"Run directory đã tồn tại: {run_dir}")
    indexed_rows = list(enumerate(rows))
    print(f"Inference: {selected_gpus} GPU, {worker_count} worker(s), batch_size={batch_size}/worker", flush=True)
    if worker_count == 2:
        # CUDA requires spawn rather than fork. Do not load the model in the parent.
        with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(
                _generate_shard, indexed_rows[device_id::2], generator_options,
                selected_top_k, selected_seed, device_id=device_id,
                generator_factory=generator_factory, batch_size=batch_size,
            ) for device_id in range(2)]
            shards = [future.result() for future in futures]
    else:
        shards = [_generate_shard(
            indexed_rows, generator_options, selected_top_k, selected_seed,
            device_id=0 if selected_gpus else None, generator_factory=generator_factory, batch_size=batch_size,
        )]
    ordered = sorted(item for shard, _ in shards for item in shard)
    if [index for index, _ in ordered] != list(range(len(rows))):
        raise RuntimeError("Kết quả worker thiếu hoặc trùng chỉ số input")
    details = [detail for _, detail in ordered]
    model_metadata = shards[0][1]
    results = [{"id": detail["id"], "answer": detail["answer"]} for detail in details]

    submission = build_submission(results)
    scores = score_if_available(
        submission,
        rows,
        scorer=scoring.score_submission,
    )
    finished = datetime.now(timezone.utc)
    manifest = json.loads(
        (version_root / "version.json").read_text(encoding="utf-8")
    )
    metrics = {
        "schema_version": 1,
        "run_id": identifier,
        "status": "completed",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": round((finished - started).total_seconds(), 3),
        "input_json": str(selected_input),
        "input_sha256": _sha256(selected_input),
        "output_answers": str(run_dir / "answers.json"),
        "model_version": config["model_version"],
        "version_manifest": manifest,
        "model_key": model_key,
        "model_id": model_config["model_id"],
        "adapter_path": model_metadata["adapter_path"],
        "parameter_count": model_metadata["parameter_count"],
        "gpu_count": selected_gpus,
        "worker_count": worker_count,
        "batch_size_per_worker": batch_size,
        "sample_latency_mode": "amortized_batch" if batch_size > 1 and config["decoding_mode"] == "greedy" else "serial",
        "timing_includes_model_loading": True,
        "prompt_mode": config["prompt_mode"],
        "decoding_mode": config["decoding_mode"],
        "quality_retry_enabled": config["enable_quality_retry"],
        "top_k": selected_top_k,
        "seed": selected_seed,
        "sample_count": len(rows),
        "format_valid_rate": _mean_flag(details, "format_valid"),
        "answer_complete_rate": _mean_flag(details, "answer_complete"),
        "retry_rate": _mean_flag(details, "retry_used"),
        "quality_retry_rate": _mean_flag(details, "quality_retry_used"),
        "quality_retry_acceptance_rate": _mean_flag(
            details, "quality_retry_accepted"
        ),
        "token_limit_retry_rate": _mean_flag(details, "token_limit_retry_used"),
        **scores,
    }
    _write_json(run_dir / "answers.json", submission)
    _write_json(run_dir / "run_metrics.json", metrics)
    _write_jsonl(run_dir / "details.jsonl", details)
    return run_dir
