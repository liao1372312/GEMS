#!/usr/bin/env python
"""Evaluate retrieval scoring ablations on endpoint selection tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gems.data import ProcessedDataset
from gems.graph_memory import ExecutionGraphMemory, endpoint_api_node_id
from gems.retrieval import ROLE_TYPE_PRIORS, RoleSpecificRetriever, node_score


@dataclass
class Metrics:
    top1: float
    top3: float
    mrr: float
    count: int


ScoreFn = Callable[[str, str], float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="dataset/processed")
    parser.add_argument("--memory", default="outputs/gems_memory.json")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--role", default="provider", choices=["planner", "provider", "executor", "supervisor"])
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--output", default="outputs/retrieval_ablation_eval.json")
    return parser.parse_args()


def stable_random_score(query: str, node_id: str) -> float:
    digest = hashlib.sha1(f"{query}\t{node_id}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def evaluate_variant(
    memory: ExecutionGraphMemory,
    retriever: RoleSpecificRetriever,
    tasks: list[dict[str, Any]],
    role: str,
    score_variant: str,
) -> tuple[Metrics, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    hits1 = 0
    hits3 = 0
    reciprocal_sum = 0.0
    evaluated = 0
    priors = ROLE_TYPE_PRIORS.get(role, {})

    for task in tasks:
        candidate_node_ids = [
            endpoint_api_node_id(endpoint_id)
            for endpoint_id in task.get("candidate_endpoint_ids", [])
            if endpoint_api_node_id(endpoint_id) in memory.nodes
        ]
        positive_node_id = endpoint_api_node_id(task["positive_endpoint_id"])
        if positive_node_id not in candidate_node_ids:
            continue

        query = task["query"]
        similarities = retriever.text_index.score_subset(retriever.role_query(query, role), candidate_node_ids)
        risk_trigger = False
        if score_variant == "gems_selective":
            ranked_by_text = sorted(
                candidate_node_ids,
                key=lambda item: similarities.get(item, 0.0),
                reverse=True,
            )
            if len(ranked_by_text) >= 2:
                top_sim = similarities.get(ranked_by_text[0], 0.0)
                second_sim = similarities.get(ranked_by_text[1], 0.0)
                margin = top_sim - second_sim
            else:
                margin = 1.0
            top_node = memory.nodes[ranked_by_text[0]] if ranked_by_text else None
            top_risk = float(top_node.risk) if top_node else 0.0
            top_conflict = float(top_node.conflict) if top_node else 0.0
            # Clean endpoint matching is near-saturated by semantics. GEMS only
            # activates reliability evidence when the semantic winner carries a
            # strong explicit risk/conflict signal; otherwise it preserves the
            # semantic/no-reliability ranking that works best on clean endpoint
            # retrieval. Ambiguity alone is not enough, because it over-triggers
            # reliability on ordinary endpoint matching.
            _ = margin
            risk_trigger = top_risk >= 0.95 or top_conflict >= 0.95

        scores: dict[str, float] = {}
        for node_id in candidate_node_ids:
            node = memory.nodes[node_id]
            sim = similarities.get(node_id, 0.0)
            rel = float(node.reliability)
            type_prior = float(priors.get(node.node_type, 0.0))
            risk = float(node.risk)
            conflict = float(node.conflict)
            initial_confidence = float(node.attrs.get("initial_confidence", 0.5) or 0.5)

            if score_variant == "gems_full":
                score = node_score(
                    similarity=sim,
                    reliability=rel,
                    type_prior=type_prior,
                    risk=risk,
                    conflict=conflict,
                )
            elif score_variant == "reliability_intent":
                score = node_score(
                    similarity=sim,
                    reliability=rel,
                    type_prior=type_prior,
                    risk=risk,
                    conflict=conflict,
                    reliability_intent=True,
                )
            elif score_variant == "text_only":
                score = sim
            elif score_variant == "reliability_only":
                score = rel
            elif score_variant == "initial_confidence_only":
                score = initial_confidence
            elif score_variant == "text_plus_initial_confidence":
                score = 0.60 * sim + 0.40 * initial_confidence
            elif score_variant == "text_plus_reliability":
                score = 0.60 * sim + 0.40 * rel
            elif score_variant == "no_reliability":
                score = 0.48 * sim + 0.16 * type_prior - 0.12 * risk - 0.10 * conflict
            elif score_variant == "gems_selective":
                if risk_trigger:
                    score = node_score(
                        similarity=sim,
                        reliability=rel,
                        type_prior=type_prior,
                        risk=risk,
                        conflict=conflict,
                        reliability_intent=True,
                    )
                else:
                    score = 0.48 * sim + 0.16 * type_prior - 0.12 * risk - 0.10 * conflict
            elif score_variant == "no_type_prior":
                score = 0.48 * sim + 0.32 * rel - 0.12 * risk - 0.10 * conflict
            elif score_variant == "no_risk_conflict":
                score = 0.48 * sim + 0.32 * rel + 0.16 * type_prior
            elif score_variant == "deterministic_random":
                score = stable_random_score(query, node_id)
            else:
                raise ValueError(f"Unknown score variant: {score_variant}")
            scores[node_id] = score

        ranked = sorted(candidate_node_ids, key=lambda node_id: scores.get(node_id, 0.0), reverse=True)
        rank = ranked.index(positive_node_id) + 1
        evaluated += 1
        hits1 += int(rank == 1)
        hits3 += int(rank <= 3)
        reciprocal_sum += 1.0 / rank
        rows.append(
            {
                "task_id": task["task_id"],
                "query": query,
                "positive_endpoint_id": task["positive_endpoint_id"],
                "rank": rank,
                "top_endpoint_id": memory.nodes[ranked[0]].attrs.get("endpoint_id") if ranked else None,
            }
        )

    if evaluated == 0:
        return Metrics(0.0, 0.0, 0.0, 0), rows
    return Metrics(hits1 / evaluated, hits3 / evaluated, reciprocal_sum / evaluated, evaluated), rows


def main() -> None:
    args = parse_args()
    dataset = ProcessedDataset.load(args.data_dir)
    memory = ExecutionGraphMemory.load(args.memory)
    retriever = RoleSpecificRetriever(memory)
    tasks = list(dataset.tasks(args.split))
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    variants = [
        "gems_selective",
        "gems_full",
        "reliability_intent",
        "text_only",
        "reliability_only",
        "initial_confidence_only",
        "text_plus_initial_confidence",
        "text_plus_reliability",
        "no_reliability",
        "no_type_prior",
        "no_risk_conflict",
        "deterministic_random",
    ]
    results: dict[str, Any] = {}
    for variant in variants:
        metrics, rows = evaluate_variant(memory, retriever, tasks, args.role, variant)
        results[variant] = {
            "metrics": metrics.__dict__,
            "examples": rows[:20],
        }

    payload = {
        "split": args.split,
        "role": args.role,
        "num_tasks": len(tasks),
        "variants": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"{'variant':32s} {'top1':>8s} {'top3':>8s} {'mrr':>8s} {'count':>8s}")
    for variant in variants:
        metrics = results[variant]["metrics"]
        print(
            f"{variant:32s} "
            f"{metrics['top1']:8.4f} {metrics['top3']:8.4f} {metrics['mrr']:8.4f} {metrics['count']:8d}"
        )
    print(f"saved {output}")


if __name__ == "__main__":
    main()
