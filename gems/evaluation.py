"""Evaluation helpers for endpoint selection tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .graph_memory import ExecutionGraphMemory, endpoint_api_node_id
from .retrieval import RoleSpecificRetriever


@dataclass
class RankingMetrics:
    top1: float
    top3: float
    mrr: float
    count: int


def evaluate_endpoint_ranking(
    memory: ExecutionGraphMemory,
    tasks: list[dict[str, Any]],
    role: str = "provider",
) -> tuple[RankingMetrics, list[dict[str, Any]]]:
    retriever = RoleSpecificRetriever(memory)
    rows: list[dict[str, Any]] = []
    hits1 = 0
    hits3 = 0
    reciprocal_sum = 0.0
    evaluated = 0
    for task in tasks:
        candidate_node_ids = [
            endpoint_api_node_id(endpoint_id)
            for endpoint_id in task.get("candidate_endpoint_ids", [])
            if endpoint_api_node_id(endpoint_id) in memory.nodes
        ]
        positive_node_id = endpoint_api_node_id(task["positive_endpoint_id"])
        if positive_node_id not in candidate_node_ids:
            continue
        scores = retriever.score_nodes(task["query"], role, candidate_node_ids)
        ranked = sorted(candidate_node_ids, key=lambda node_id: scores.get(node_id, 0.0), reverse=True)
        rank = ranked.index(positive_node_id) + 1
        evaluated += 1
        hits1 += int(rank == 1)
        hits3 += int(rank <= 3)
        reciprocal_sum += 1.0 / rank
        rows.append(
            {
                "task_id": task["task_id"],
                "query": task["query"],
                "positive_endpoint_id": task["positive_endpoint_id"],
                "rank": rank,
                "top_endpoint_id": memory.nodes[ranked[0]].attrs.get("endpoint_id") if ranked else None,
                "scores": {
                    memory.nodes[node_id].attrs.get("endpoint_id", node_id): round(scores[node_id], 6)
                    for node_id in ranked
                },
            }
        )
    if evaluated == 0:
        return RankingMetrics(0.0, 0.0, 0.0, 0), rows
    return RankingMetrics(hits1 / evaluated, hits3 / evaluated, reciprocal_sum / evaluated, evaluated), rows
