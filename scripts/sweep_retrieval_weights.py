#!/usr/bin/env python
"""Sweep text/reliability weights across selection and reliability tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_reliability_stress import build_groups
from gems.data import ProcessedDataset
from gems.graph_memory import ExecutionGraphMemory, endpoint_api_node_id
from gems.retrieval import RoleSpecificRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="dataset/processed")
    parser.add_argument("--memory", default="outputs/gems_memory_train_only.json")
    parser.add_argument("--role", default="provider")
    parser.add_argument("--output", default="outputs/retrieval_weight_sweep.json")
    return parser.parse_args()


def rank_selection(
    memory: ExecutionGraphMemory,
    retriever: RoleSpecificRetriever,
    tasks: list[dict[str, Any]],
    role: str,
    sim_weight: float,
    rel_weight: float,
) -> dict[str, float]:
    hits1 = 0
    hits3 = 0
    reciprocal_sum = 0.0
    count = 0
    for task in tasks:
        node_ids = [
            endpoint_api_node_id(endpoint_id)
            for endpoint_id in task.get("candidate_endpoint_ids", [])
            if endpoint_api_node_id(endpoint_id) in memory.nodes
        ]
        positive = endpoint_api_node_id(task["positive_endpoint_id"])
        if positive not in node_ids:
            continue
        sims = retriever.text_index.score_subset(retriever.role_query(task["query"], role), node_ids)
        scores = {
            node_id: sim_weight * sims.get(node_id, 0.0) + rel_weight * memory.nodes[node_id].reliability
            for node_id in node_ids
        }
        ranked = sorted(node_ids, key=lambda node_id: scores[node_id], reverse=True)
        rank = ranked.index(positive) + 1
        count += 1
        hits1 += int(rank == 1)
        hits3 += int(rank <= 3)
        reciprocal_sum += 1.0 / rank
    return {
        "top1": hits1 / count if count else 0.0,
        "top3": hits3 / count if count else 0.0,
        "mrr": reciprocal_sum / count if count else 0.0,
        "count": count,
    }


def rank_stress(
    memory: ExecutionGraphMemory,
    retriever: RoleSpecificRetriever,
    groups: list[dict[str, Any]],
    role: str,
    sim_weight: float,
    rel_weight: float,
) -> dict[str, float]:
    hits1 = 0
    hits3 = 0
    reciprocal_sum = 0.0
    count = 0
    for group in groups:
        node_ids = [
            endpoint_api_node_id(endpoint_id)
            for endpoint_id in group["candidate_endpoint_ids"]
            if endpoint_api_node_id(endpoint_id) in memory.nodes
        ]
        success_ids = {
            endpoint_api_node_id(endpoint_id)
            for endpoint_id in group["successful_endpoint_ids"]
            if endpoint_api_node_id(endpoint_id) in memory.nodes
        }
        if not node_ids or not success_ids:
            continue
        sims = retriever.text_index.score_subset(retriever.role_query(group["query"], role), node_ids)
        scores = {
            node_id: sim_weight * sims.get(node_id, 0.0) + rel_weight * memory.nodes[node_id].reliability
            for node_id in node_ids
        }
        ranked = sorted(node_ids, key=lambda node_id: scores[node_id], reverse=True)
        first_success_rank = next(
            (rank for rank, node_id in enumerate(ranked, start=1) if node_id in success_ids),
            None,
        )
        if first_success_rank is None:
            continue
        count += 1
        hits1 += int(first_success_rank == 1)
        hits3 += int(first_success_rank <= 3)
        reciprocal_sum += 1.0 / first_success_rank
    return {
        "success_at_1": hits1 / count if count else 0.0,
        "success_at_3": hits3 / count if count else 0.0,
        "mrr_first_success": reciprocal_sum / count if count else 0.0,
        "count": count,
    }


def main() -> None:
    args = parse_args()
    dataset = ProcessedDataset.load(args.data_dir)
    memory = ExecutionGraphMemory.load(args.memory)
    retriever = RoleSpecificRetriever(memory)
    val_tasks = list(dataset.tasks("val"))
    test_tasks = list(dataset.tasks("test"))
    val_groups = build_groups(dataset.endpoints, "val", success_k=4, failed_k=3, max_groups=0)
    test_groups = build_groups(dataset.endpoints, "test", success_k=4, failed_k=3, max_groups=0)

    weights = [
        (1.00, 0.00),
        (0.95, 0.05),
        (0.90, 0.10),
        (0.80, 0.20),
        (0.70, 0.30),
        (0.60, 0.40),
        (0.50, 0.50),
        (0.40, 0.60),
        (0.30, 0.70),
        (0.20, 0.80),
        (0.10, 0.90),
        (0.00, 1.00),
    ]
    rows: list[dict[str, Any]] = []
    for sim_weight, rel_weight in weights:
        val_selection = rank_selection(memory, retriever, val_tasks, args.role, sim_weight, rel_weight)
        val_stress = rank_stress(memory, retriever, val_groups, args.role, sim_weight, rel_weight)
        test_selection = rank_selection(memory, retriever, test_tasks, args.role, sim_weight, rel_weight)
        test_stress = rank_stress(memory, retriever, test_groups, args.role, sim_weight, rel_weight)
        val_balanced = (val_selection["top1"] + val_stress["success_at_1"]) / 2.0
        test_balanced = (test_selection["top1"] + test_stress["success_at_1"]) / 2.0
        rows.append(
            {
                "sim_weight": sim_weight,
                "rel_weight": rel_weight,
                "val_balanced_top1": val_balanced,
                "test_balanced_top1": test_balanced,
                "val_selection": val_selection,
                "val_stress": val_stress,
                "test_selection": test_selection,
                "test_stress": test_stress,
            }
        )

    best_val = max(rows, key=lambda row: row["val_balanced_top1"])
    payload = {"best_by_val_balanced_top1": best_val, "rows": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"{'sim':>5s} {'rel':>5s} {'val_sel':>8s} {'val_stress':>10s} {'test_sel':>9s} {'test_stress':>11s} {'val_bal':>8s}")
    for row in rows:
        print(
            f"{row['sim_weight']:5.2f} {row['rel_weight']:5.2f} "
            f"{row['val_selection']['top1']:8.4f} {row['val_stress']['success_at_1']:10.4f} "
            f"{row['test_selection']['top1']:9.4f} {row['test_stress']['success_at_1']:11.4f} "
            f"{row['val_balanced_top1']:8.4f}"
        )
    print("best_by_val", json.dumps(best_val, ensure_ascii=False, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
