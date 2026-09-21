from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any


def _official_dependencies():
    import numpy as np
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    return np, meteor_score, rouge_scorer


def score_text_pairs(predictions: list[str], references: list[str]) -> dict[str, Any]:
    if len(predictions) != len(references):
        raise ValueError("Số prediction không khớp số reference")
    _, meteor_score, rouge_scorer = _official_dependencies()
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    per_sample = []
    for prediction, reference in zip(predictions, references):
        rouge_l = rouge.score(str(reference), str(prediction))["rougeL"].fmeasure
        meteor = meteor_score([str(reference).split()], str(prediction).split())
        per_sample.append({"rouge": float(rouge_l), "meteor": float(meteor)})
    return {
        "rouge": statistics.fmean(item["rouge"] for item in per_sample),
        "meteor": statistics.fmean(item["meteor"] for item in per_sample),
        "count": len(per_sample),
        "per_sample": per_sample,
        "tokenizer": "python str.split (official)",
    }


def score_submission(prediction: dict, reference: dict) -> dict[str, Any]:
    predicted_answers = {key: value["answer"] for key, value in prediction.items()}
    if len(predicted_answers) != len(reference):
        raise ValueError("Samples in predict not match with reference")
    missing = set(predicted_answers) - set(reference)
    if missing:
        raise KeyError(f"Prediction có ID không tồn tại trong reference: {sorted(missing)[:5]}")
    ids = list(predicted_answers)
    return score_text_pairs(
        [predicted_answers[key] for key in ids],
        [reference[key] for key in ids],
    )

