#!/usr/bin/env python
"""Analyze where full-test LLM reranking helps or hurts semantic top-1."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_public_composition_rerank import selected_index_from_result
from run_public_composition_experiments import load_composition_examples, split_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--llm-result", default="outputs/llm_public_composition_rerank_test_all.json")
    parser.add_argument("--output", default="outputs/llm_vs_semantic_analysis.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def key(record_id: str, step_id: int) -> str:
    return f"{record_id}::{step_id}"


def main() -> None:
    args = parse_args()
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    test_examples = [example for example in examples if example.record_id in splits["test"]]
    llm = json.loads(Path(args.llm_result).read_text(encoding="utf-8"))
    predictions = {
        key(str(pred["record_id"]), int(pred["step_id"])): pred
        for pred in llm.get("predictions", [])
    }

    counts = Counter()
    selected_rank_when_help = Counter()
    selected_rank_when_hurt = Counter()
    confidence_help = []
    confidence_hurt = []
    examples_out = {"helped": [], "hurt": [], "both_wrong": []}

    for example in test_examples:
        pred = predictions.get(key(example.record_id, example.step_id))
        if not pred:
            continue
        semantic_correct = example.gold_index == 0
        llm_index = selected_index_from_result(pred, len(example.candidates))
        llm_correct = llm_index == example.gold_index
        parsed = pred.get("parsed") or {}
        confidence = float(parsed.get("confidence") or 0.0)
        if semantic_correct and llm_correct:
            counts["both_correct"] += 1
        elif semantic_correct and not llm_correct:
            counts["llm_hurt"] += 1
            selected_rank_when_hurt[llm_index + 1] += 1
            confidence_hurt.append(confidence)
            if len(examples_out["hurt"]) < 10:
                examples_out["hurt"].append(
                    {
                        "record_id": example.record_id,
                        "step_id": example.step_id,
                        "gold_rank": example.gold_index + 1,
                        "llm_rank": llm_index + 1,
                        "confidence": confidence,
                        "rationale": parsed.get("rationale"),
                    }
                )
        elif not semantic_correct and llm_correct:
            counts["llm_helped"] += 1
            selected_rank_when_help[llm_index + 1] += 1
            confidence_help.append(confidence)
            if len(examples_out["helped"]) < 10:
                examples_out["helped"].append(
                    {
                        "record_id": example.record_id,
                        "step_id": example.step_id,
                        "gold_rank": example.gold_index + 1,
                        "llm_rank": llm_index + 1,
                        "confidence": confidence,
                        "rationale": parsed.get("rationale"),
                    }
                )
        else:
            counts["both_wrong"] += 1
            if len(examples_out["both_wrong"]) < 10:
                examples_out["both_wrong"].append(
                    {
                        "record_id": example.record_id,
                        "step_id": example.step_id,
                        "gold_rank": example.gold_index + 1,
                        "llm_rank": llm_index + 1,
                        "confidence": confidence,
                        "rationale": parsed.get("rationale"),
                    }
                )

    total = sum(counts.values())
    semantic_acc = (counts["both_correct"] + counts["llm_hurt"]) / total if total else 0.0
    llm_acc = (counts["both_correct"] + counts["llm_helped"]) / total if total else 0.0
    payload: dict[str, Any] = {
        "total": total,
        "counts": dict(counts),
        "semantic_acc": semantic_acc,
        "llm_acc": llm_acc,
        "net_api_acc_gain": llm_acc - semantic_acc,
        "llm_helped_rate": counts["llm_helped"] / total if total else 0.0,
        "llm_hurt_rate": counts["llm_hurt"] / total if total else 0.0,
        "selected_rank_when_help": dict(sorted(selected_rank_when_help.items())),
        "selected_rank_when_hurt": dict(sorted(selected_rank_when_hurt.items())),
        "avg_confidence_help": sum(confidence_help) / len(confidence_help) if confidence_help else 0.0,
        "avg_confidence_hurt": sum(confidence_hurt) / len(confidence_hurt) if confidence_hurt else 0.0,
        "examples": examples_out,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
