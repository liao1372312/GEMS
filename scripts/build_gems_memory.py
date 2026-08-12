#!/usr/bin/env python
"""Build a serialized GEMS graph memory from ``dataset/processed``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gems.data import ProcessedDataset
from gems.graph_memory import ExecutionGraphMemory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="dataset/processed")
    parser.add_argument("--output", default="outputs/gems_memory.json")
    parser.add_argument("--include-val-test-tasks", action="store_true")
    parser.add_argument(
        "--feedback-splits",
        default="train",
        help="Comma-separated feedback splits inserted as historical execution evidence; use 'all' for every split.",
    )
    parser.add_argument("--propagation-layers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ProcessedDataset.load(args.data_dir)
    feedback_splits = {part.strip() for part in args.feedback_splits.split(",") if part.strip()}
    feedback_events = dataset.feedback_events
    if feedback_splits and "all" not in feedback_splits:
        feedback_events = [event for event in dataset.feedback_events if event.get("split") in feedback_splits]
    memory = ExecutionGraphMemory.from_processed_dataset(
        dataset.services,
        dataset.endpoints,
        feedback_events,
        dataset.selection_tasks,
        train_only_tasks=not args.include_val_test_tasks,
    )
    memory.propagate_reliability(layers=args.propagation_layers)
    memory.save(args.output)
    print(
        f"saved {args.output}: {len(memory.nodes)} nodes, "
        f"{len(memory.edges)} edges, {args.propagation_layers} propagation layers, "
        f"feedback_splits={args.feedback_splits}"
    )


if __name__ == "__main__":
    main()
