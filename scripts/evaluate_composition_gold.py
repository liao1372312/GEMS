#!/usr/bin/env python
"""Evaluate multi-step composition gold selections."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Metrics:
    step_top1: float
    step_top3: float
    step_mrr: float
    workflow_exact_top1: float
    avg_selected_rank: float
    records: int
    steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--output", default="outputs/composition_gold_eval.json")
    return parser.parse_args()


def normalize_domain(value: object) -> str:
    text = str(value or "unknown").strip().lower()
    return text or "unknown"


def selected_rank(step: dict[str, Any]) -> int | None:
    selection = step.get("selection") or {}
    rank = selection.get("selected_rank")
    if isinstance(rank, int):
        return rank
    try:
        return int(rank)
    except (TypeError, ValueError):
        return None


def record_domain(record: dict[str, Any]) -> str:
    return normalize_domain(record.get("domain") or (record.get("metadata") or {}).get("paper_domain"))


def evaluate(records: list[dict[str, Any]]) -> tuple[Metrics, dict[str, Any]]:
    step_ranks: list[int] = []
    workflow_top1_hits = 0
    domain_ranks: dict[str, list[int]] = defaultdict(list)
    task_len_counts: Counter[int] = Counter()
    selected_rank_counts: Counter[int] = Counter()
    fallback_steps = 0
    confidence_values: list[float] = []
    dependency_records = 0

    for record in records:
        domain = record_domain(record)
        steps = record.get("gold_api_candidates") or []
        task_len_counts[len(steps)] += 1
        ranks_for_record: list[int] = []
        dependency = record.get("dependencies", record.get("dependency"))
        if dependency:
            dependency_records += 1

        for step in steps:
            rank = selected_rank(step)
            if rank is None:
                continue
            ranks_for_record.append(rank)
            step_ranks.append(rank)
            selected_rank_counts[rank] += 1
            domain_ranks[domain].append(rank)
            selection = step.get("selection") or {}
            confidence_values.append(float(selection.get("confidence") or 0.0))
            if "Fallback after LLM error" in str(selection.get("rationale") or ""):
                fallback_steps += 1
        if ranks_for_record and all(rank == 1 for rank in ranks_for_record):
            workflow_top1_hits += 1

    steps = len(step_ranks)
    records_count = len(records)
    reciprocal_sum = sum(1.0 / rank for rank in step_ranks)
    metrics = Metrics(
        step_top1=sum(1 for rank in step_ranks if rank == 1) / steps if steps else 0.0,
        step_top3=sum(1 for rank in step_ranks if rank <= 3) / steps if steps else 0.0,
        step_mrr=reciprocal_sum / steps if steps else 0.0,
        workflow_exact_top1=workflow_top1_hits / records_count if records_count else 0.0,
        avg_selected_rank=sum(step_ranks) / steps if steps else 0.0,
        records=records_count,
        steps=steps,
    )
    diagnostics = {
        "task_length_counts": dict(sorted(task_len_counts.items())),
        "selected_rank_counts": dict(sorted(selected_rank_counts.items())),
        "fallback_steps": fallback_steps,
        "fallback_step_rate": fallback_steps / steps if steps else 0.0,
        "dependency_records": dependency_records,
        "dependency_record_rate": dependency_records / records_count if records_count else 0.0,
        "confidence_mean": sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
        "by_domain": {
            domain: {
                "steps": len(ranks),
                "step_top1": sum(1 for rank in ranks if rank == 1) / len(ranks),
                "step_top3": sum(1 for rank in ranks if rank <= 3) / len(ranks),
                "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
                "avg_selected_rank": sum(ranks) / len(ranks),
            }
            for domain, ranks in sorted(domain_ranks.items())
        },
    }
    return metrics, diagnostics


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics, diagnostics = evaluate(records)
    payload = {
        "input": str(input_path),
        "metrics": metrics.__dict__,
        "diagnostics": diagnostics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))
    print("selected_rank_counts", diagnostics["selected_rank_counts"])
    print("fallback_step_rate", round(diagnostics["fallback_step_rate"], 4))
    print("dependency_record_rate", round(diagnostics["dependency_record_rate"], 4))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
