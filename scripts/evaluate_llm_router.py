#!/usr/bin/env python
"""Evaluate a deployable semantic/LLM router using available LLM cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_public_composition_rerank import selected_index_from_result
from run_public_composition_experiments import (
    f1_score,
    load_composition_examples,
    required_param_names,
    split_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--llm-files", nargs="*", default=[])
    parser.add_argument("--output", default="outputs/llm_router_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def key(record_id: str, step_id: int) -> str:
    return f"{record_id}::{step_id}"


def load_llm_predictions(files: list[str]) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    paths = [Path(name) for name in files] if files else sorted(Path("outputs").glob("llm_public_composition_rerank*.json"))
    for path in paths:
        obj = json.loads(path.read_text(encoding="utf-8"))
        # Ignore dry-run artifacts.
        for pred in obj.get("predictions") or []:
            if pred.get("missing_cache") or pred.get("error"):
                continue
            if (pred.get("parsed") or {}).get("rationale") == "dry_run":
                continue
            predictions[key(str(pred.get("record_id")), int(pred.get("step_id")))] = pred
    return predictions


def margin(example: Any) -> float:
    if len(example.candidates) < 2:
        return 1.0
    first = float(example.candidates[0].get("similarity_score") or 0.0)
    second = float(example.candidates[1].get("similarity_score") or 0.0)
    return first - second


def route_to_llm(example: Any, threshold: float) -> bool:
    return margin(example) <= threshold


def evaluate(examples: list[Any], predictions: dict[str, dict[str, Any]], threshold: float) -> dict[str, Any]:
    hits = 0
    workflows: dict[str, list[bool]] = {}
    para_f1 = []
    llm_calls = 0
    llm_cache_hits = 0
    missing_llm = 0
    for example in examples:
        use_llm = route_to_llm(example, threshold)
        pred_index = 0
        if use_llm:
            llm_calls += 1
            pred = predictions.get(key(example.record_id, example.step_id))
            if pred:
                llm_cache_hits += 1
                pred_index = selected_index_from_result(pred, len(example.candidates))
            else:
                missing_llm += 1
        correct = pred_index == example.gold_index
        hits += int(correct)
        workflows.setdefault(example.record_id, []).append(correct)
        para_f1.append(
            f1_score(
                required_param_names(example.candidates[pred_index]),
                required_param_names(example.gold_candidate),
            )
        )
    n = len(examples)
    return {
        "api_acc": hits / n if n else 0.0,
        "workflow_exact": sum(1 for values in workflows.values() if all(values)) / len(workflows) if workflows else 0.0,
        "para_f1": sum(para_f1) / len(para_f1) if para_f1 else 0.0,
        "llm_call_rate": llm_calls / n if n else 0.0,
        "llm_calls": llm_calls,
        "llm_cache_hits": llm_cache_hits,
        "missing_llm": missing_llm,
        "coverage": llm_cache_hits / llm_calls if llm_calls else 1.0,
        "steps": n,
    }


def main() -> None:
    args = parse_args()
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    split_examples = {
        split: [example for example in examples if example.record_id in ids]
        for split, ids in splits.items()
    }
    predictions = load_llm_predictions(args.llm_files)
    thresholds = sorted({0.0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20})
    val_rows = [{"threshold": t, "metrics": evaluate(split_examples["val"], predictions, t)} for t in thresholds]
    val_full_coverage = all(row["metrics"]["missing_llm"] == 0 for row in val_rows)
    # Prefer high validation accuracy; lightly penalize LLM call rate.
    best = max(val_rows, key=lambda row: (row["metrics"]["api_acc"] - 0.02 * row["metrics"]["llm_call_rate"], row["metrics"]["workflow_exact"]))
    test = evaluate(split_examples["test"], predictions, best["threshold"])
    semantic = evaluate(split_examples["test"], predictions, -1.0)
    payload = {
        "paper_ready": val_full_coverage,
        "best_threshold": best["threshold"],
        "val": best["metrics"],
        "test": test,
        "semantic_top1_test": semantic,
        "available_llm_predictions": len(predictions),
        "threshold_sweep": val_rows,
        "note": (
            "paper-ready: validation routed examples have full LLM coverage"
            if val_full_coverage
            else "diagnostic only: validation routed examples have missing LLM predictions; run full/all validation LLM eval to complete deployable router measurement."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
