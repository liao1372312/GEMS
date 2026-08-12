#!/usr/bin/env python
"""Reliability stress test with semantically hard failed endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gems.data import ProcessedDataset
from gems.graph_memory import ExecutionGraphMemory, endpoint_api_node_id
from gems.retrieval import ROLE_TYPE_PRIORS, RoleSpecificRetriever, node_score
from gems.text import normalize_text


@dataclass
class Metrics:
    success_at_1: float
    success_at_3: float
    mrr_first_success: float
    avg_top1_observed_success: float
    count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="dataset/processed")
    parser.add_argument("--memory", default="outputs/gems_memory.json")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--role", default="provider", choices=["planner", "provider", "executor", "supervisor"])
    parser.add_argument("--success-candidates", type=int, default=4)
    parser.add_argument("--failed-candidates", type=int, default=3)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--output", default="outputs/reliability_stress_eval.json")
    return parser.parse_args()


def endpoint_text(endpoint: dict[str, Any]) -> str:
    return normalize_text(
        ". ".join(
            str(part)
            for part in [
                endpoint.get("interface_text"),
                endpoint.get("endpoint_description"),
                endpoint.get("service_description"),
                endpoint.get("endpoint_name"),
                endpoint.get("category"),
            ]
            if part
        )
    )


def stable_random_score(query: str, node_id: str) -> float:
    digest = hashlib.sha1(f"{query}\t{node_id}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def build_groups(
    endpoints: list[dict[str, Any]],
    split: str,
    success_k: int,
    failed_k: int,
    max_groups: int,
) -> list[dict[str, Any]]:
    split_endpoints = [endpoint for endpoint in endpoints if endpoint.get("split") == split]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for endpoint in split_endpoints:
        if endpoint.get("observed_success") is not None:
            by_category[endpoint.get("category") or "unknown"].append(endpoint)

    groups: list[dict[str, Any]] = []
    for category in sorted(by_category):
        records = by_category[category]
        success = [record for record in records if record.get("observed_success") is True]
        failed = [record for record in records if record.get("observed_success") is False]
        if not success or not failed:
            continue

        docs = [endpoint_text(record) or "empty" for record in records]
        matrix = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2)).fit_transform(docs)
        record_index = {record["endpoint_id"]: idx for idx, record in enumerate(records)}

        for anchor in sorted(failed, key=lambda item: item["endpoint_id"]):
            anchor_idx = record_index[anchor["endpoint_id"]]
            sims = cosine_similarity(matrix[anchor_idx], matrix).ravel()

            success_ranked = sorted(
                success,
                key=lambda item: (sims[record_index[item["endpoint_id"]]], item["endpoint_id"]),
                reverse=True,
            )[:success_k]
            failed_ranked = [
                item
                for item in sorted(
                    failed,
                    key=lambda item: (sims[record_index[item["endpoint_id"]]], item["endpoint_id"]),
                    reverse=True,
                )
                if item["endpoint_id"] != anchor["endpoint_id"]
            ][: max(0, failed_k - 1)]

            candidates = [anchor] + success_ranked + failed_ranked
            if not success_ranked:
                continue
            query = f"Find a reliable web API endpoint similar to: {endpoint_text(anchor)}"
            groups.append(
                {
                    "anchor_failed_endpoint_id": anchor["endpoint_id"],
                    "category": category,
                    "query": query,
                    "candidate_endpoint_ids": [record["endpoint_id"] for record in candidates],
                    "successful_endpoint_ids": [record["endpoint_id"] for record in candidates if record.get("observed_success") is True],
                    "failed_endpoint_ids": [record["endpoint_id"] for record in candidates if record.get("observed_success") is False],
                }
            )
            if max_groups and len(groups) >= max_groups:
                return groups
    return groups


def score_variant(
    memory: ExecutionGraphMemory,
    retriever: RoleSpecificRetriever,
    query: str,
    role: str,
    node_ids: list[str],
    variant: str,
) -> dict[str, float]:
    similarities = retriever.text_index.score_subset(retriever.role_query(query, role), node_ids)
    priors = ROLE_TYPE_PRIORS.get(role, {})
    scores: dict[str, float] = {}
    for node_id in node_ids:
        node = memory.nodes[node_id]
        sim = similarities.get(node_id, 0.0)
        rel = float(node.reliability)
        type_prior = float(priors.get(node.node_type, 0.0))
        risk = float(node.risk)
        conflict = float(node.conflict)
        initial_confidence = float(node.attrs.get("initial_confidence", 0.5) or 0.5)

        if variant == "gems_full":
            score = node_score(
                similarity=sim,
                reliability=rel,
                type_prior=type_prior,
                risk=risk,
                conflict=conflict,
            )
        elif variant == "reliability_intent":
            score = node_score(
                similarity=sim,
                reliability=rel,
                type_prior=type_prior,
                risk=risk,
                conflict=conflict,
                reliability_intent=True,
            )
        elif variant == "text_only":
            score = sim
        elif variant == "reliability_only":
            score = rel
        elif variant == "initial_confidence_only":
            score = initial_confidence
        elif variant == "text_plus_initial_confidence":
            score = 0.60 * sim + 0.40 * initial_confidence
        elif variant == "text_plus_reliability":
            score = 0.60 * sim + 0.40 * rel
        elif variant == "no_reliability":
            score = 0.48 * sim + 0.16 * type_prior - 0.12 * risk - 0.10 * conflict
        elif variant == "no_risk_conflict":
            score = 0.48 * sim + 0.32 * rel + 0.16 * type_prior
        elif variant == "deterministic_random":
            score = stable_random_score(query, node_id)
        else:
            raise ValueError(f"Unknown score variant: {variant}")
        scores[node_id] = score
    return scores


def evaluate_groups(
    memory: ExecutionGraphMemory,
    retriever: RoleSpecificRetriever,
    groups: list[dict[str, Any]],
    role: str,
    variant: str,
) -> tuple[Metrics, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    hit1 = 0
    hit3 = 0
    reciprocal_sum = 0.0
    top1_success_sum = 0.0
    evaluated = 0

    for group in groups:
        node_ids = [
            endpoint_api_node_id(endpoint_id)
            for endpoint_id in group["candidate_endpoint_ids"]
            if endpoint_api_node_id(endpoint_id) in memory.nodes
        ]
        success_node_ids = {
            endpoint_api_node_id(endpoint_id)
            for endpoint_id in group["successful_endpoint_ids"]
            if endpoint_api_node_id(endpoint_id) in memory.nodes
        }
        if not node_ids or not success_node_ids:
            continue

        scores = score_variant(memory, retriever, group["query"], role, node_ids, variant)
        ranked = sorted(node_ids, key=lambda node_id: scores.get(node_id, 0.0), reverse=True)
        first_success_rank = next(
            (rank for rank, node_id in enumerate(ranked, start=1) if node_id in success_node_ids),
            None,
        )
        if first_success_rank is None:
            continue

        evaluated += 1
        top_is_success = ranked[0] in success_node_ids
        hit1 += int(top_is_success)
        hit3 += int(first_success_rank <= 3)
        reciprocal_sum += 1.0 / first_success_rank
        top1_success_sum += float(top_is_success)
        rows.append(
            {
                "anchor_failed_endpoint_id": group["anchor_failed_endpoint_id"],
                "category": group["category"],
                "first_success_rank": first_success_rank,
                "top_endpoint_id": memory.nodes[ranked[0]].attrs.get("endpoint_id"),
                "top_observed_success": top_is_success,
                "scores": {
                    memory.nodes[node_id].attrs.get("endpoint_id", node_id): round(scores[node_id], 6)
                    for node_id in ranked
                },
            }
        )

    if evaluated == 0:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0), rows
    return (
        Metrics(
            success_at_1=hit1 / evaluated,
            success_at_3=hit3 / evaluated,
            mrr_first_success=reciprocal_sum / evaluated,
            avg_top1_observed_success=top1_success_sum / evaluated,
            count=evaluated,
        ),
        rows,
    )


def main() -> None:
    args = parse_args()
    dataset = ProcessedDataset.load(args.data_dir)
    memory = ExecutionGraphMemory.load(args.memory)
    retriever = RoleSpecificRetriever(memory)
    groups = build_groups(
        dataset.endpoints,
        args.split,
        success_k=args.success_candidates,
        failed_k=args.failed_candidates,
        max_groups=args.max_groups,
    )

    variants = [
        "gems_full",
        "reliability_intent",
        "text_only",
        "reliability_only",
        "initial_confidence_only",
        "text_plus_initial_confidence",
        "text_plus_reliability",
        "no_reliability",
        "no_risk_conflict",
        "deterministic_random",
    ]
    results: dict[str, Any] = {}
    for variant in variants:
        metrics, rows = evaluate_groups(memory, retriever, groups, args.role, variant)
        results[variant] = {"metrics": metrics.__dict__, "examples": rows[:20]}

    payload = {
        "split": args.split,
        "role": args.role,
        "num_groups": len(groups),
        "group_construction": {
            "anchor": "failed endpoint",
            "query": "anchor endpoint text prefixed with reliable endpoint request",
            "success_candidates": args.success_candidates,
            "failed_candidates": args.failed_candidates,
        },
        "variants": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"groups: {len(groups)}")
    print(f"{'variant':32s} {'s@1':>8s} {'s@3':>8s} {'mrr':>8s} {'count':>8s}")
    for variant in variants:
        metrics = results[variant]["metrics"]
        print(
            f"{variant:32s} "
            f"{metrics['success_at_1']:8.4f} {metrics['success_at_3']:8.4f} "
            f"{metrics['mrr_first_success']:8.4f} {metrics['count']:8d}"
        )
    print(f"saved {output}")


if __name__ == "__main__":
    main()
