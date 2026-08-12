#!/usr/bin/env python
"""Print role-specific serialized GEMS evidence for an ad-hoc request."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gems.data import ProcessedDataset
from gems.graph_memory import ExecutionGraphMemory
from gems.retrieval import RoleSpecificRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--data-dir", default="dataset/processed")
    parser.add_argument("--memory", default="")
    parser.add_argument("--role", default="provider", choices=["planner", "provider", "executor", "supervisor"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed-top-k", type=int, default=20)
    parser.add_argument("--hops", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.memory:
        memory = ExecutionGraphMemory.load(args.memory)
    else:
        dataset = ProcessedDataset.load(args.data_dir)
        memory = ExecutionGraphMemory.from_processed_dataset(
            dataset.services,
            dataset.endpoints,
            dataset.feedback_events,
            dataset.selection_tasks,
            train_only_tasks=True,
        )
    retriever = RoleSpecificRetriever(memory)
    evidence = retriever.retrieve(
        args.query,
        args.role,
        seed_top_k=args.seed_top_k,
        hops=args.hops,
        top_k=args.top_k,
    )
    print(json.dumps(evidence.serialized, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
