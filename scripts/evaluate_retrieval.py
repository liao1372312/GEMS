#!/usr/bin/env python
"""Evaluate reliability-aware endpoint ranking on selection tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gems.data import ProcessedDataset
from gems.evaluation import evaluate_endpoint_ranking
from gems.graph_memory import ExecutionGraphMemory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="dataset/processed")
    parser.add_argument("--memory", default="")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--role", default="provider", choices=["planner", "provider", "executor", "supervisor"])
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument(
        "--feedback-splits",
        default="train",
        help="Used only when --memory is omitted. Comma-separated feedback splits or 'all'.",
    )
    parser.add_argument("--output", default="outputs/retrieval_eval.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ProcessedDataset.load(args.data_dir)
    if args.memory:
        memory = ExecutionGraphMemory.load(args.memory)
    else:
        feedback_splits = {part.strip() for part in args.feedback_splits.split(",") if part.strip()}
        feedback_events = dataset.feedback_events
        if feedback_splits and "all" not in feedback_splits:
            feedback_events = [event for event in dataset.feedback_events if event.get("split") in feedback_splits]
        memory = ExecutionGraphMemory.from_processed_dataset(
            dataset.services,
            dataset.endpoints,
            feedback_events,
            dataset.selection_tasks,
            train_only_tasks=True,
        )

    tasks = list(dataset.tasks(args.split))
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    metrics, rows = evaluate_endpoint_ranking(memory, tasks, role=args.role)
    payload = {
        "split": args.split,
        "role": args.role,
        "metrics": metrics.__dict__,
        "num_rows": len(rows),
        "examples": rows[:20],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload["metrics"], indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
