#!/usr/bin/env python
"""Export a retrieved GEMS role-specific subgraph as HTML, GraphML, and PNG."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

from gems.data import ProcessedDataset
from gems.graph_memory import ExecutionGraphMemory
from gems.retrieval import RoleSpecificRetriever
from gems.visualization import build_retrieved_nx_graph, export_graphml, export_html, export_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--memory", default="outputs/gems_memory.json")
    parser.add_argument("--data-dir", default="dataset/processed")
    parser.add_argument("--role", default="provider", choices=["planner", "provider", "executor", "supervisor"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed-top-k", type=int, default=24)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--output-prefix", default="outputs/gems_subgraph")
    parser.add_argument("--no-png", action="store_true")
    return parser.parse_args()


def load_memory(args: argparse.Namespace) -> ExecutionGraphMemory:
    memory_path = Path(args.memory)
    if memory_path.exists():
        return ExecutionGraphMemory.load(memory_path)
    dataset = ProcessedDataset.load(args.data_dir)
    return ExecutionGraphMemory.from_processed_dataset(
        dataset.services,
        dataset.endpoints,
        [event for event in dataset.feedback_events if event.get("split") == "train"],
        dataset.selection_tasks,
        train_only_tasks=True,
    )


def main() -> None:
    args = parse_args()
    memory = load_memory(args)
    retriever = RoleSpecificRetriever(memory)
    evidence = retriever.retrieve(
        args.query,
        args.role,
        seed_top_k=args.seed_top_k,
        hops=args.hops,
        top_k=args.top_k,
    )
    graph = build_retrieved_nx_graph(memory, evidence)
    prefix = Path(args.output_prefix)
    metadata = {
        "query": args.query,
        "role": args.role,
        "top_k": args.top_k,
        "seed_top_k": args.seed_top_k,
        "hops": args.hops,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "memory": args.memory,
    }
    title = f"GEMS {args.role} subgraph"
    export_html(graph, prefix.with_suffix(".html"), title=title, metadata=metadata)
    export_graphml(graph, prefix.with_suffix(".graphml"))
    if not args.no_png:
        export_png(graph, prefix.with_suffix(".png"), title=title)
    with prefix.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "evidence": evidence.serialized}, handle, ensure_ascii=False, indent=2)
    outputs = [str(prefix.with_suffix(".html")), str(prefix.with_suffix(".graphml")), str(prefix.with_suffix(".json"))]
    if not args.no_png:
        outputs.append(str(prefix.with_suffix(".png")))
    print(json.dumps({"outputs": outputs, **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
