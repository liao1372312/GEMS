#!/usr/bin/env python
"""Tune a lightweight public-composition reranker on validation data."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_public_composition_experiments import (
    ExperimentScorer,
    TraceRagScorer,
    add_composition_traces_to_memory,
    endpoint_key,
    ensure_candidate_nodes,
    evaluate_method,
    load_composition_examples,
    load_processed_endpoint_maps,
    rank_from_scores,
    split_records,
    step_text,
)
from gems.graph_memory import ExecutionGraphMemory
from gems.retrieval import RoleSpecificRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--processed-endpoints", default="dataset/processed/endpoints.jsonl")
    parser.add_argument("--memory", default="outputs/gems_memory_train_only.json")
    parser.add_argument("--output", default="outputs/public_composition_reranker_sweep.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--trace-top-k", type=int, default=16)
    return parser.parse_args()


def build_context(args: argparse.Namespace) -> dict[str, Any]:
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    train_examples = [example for example in examples if example.record_id in splits["train"]]
    val_examples = [example for example in examples if example.record_id in splits["val"]]
    test_examples = [example for example in examples if example.record_id in splits["test"]]
    endpoints_by_url, _ = load_processed_endpoint_maps(args.processed_endpoints)
    memory = ExecutionGraphMemory.load(args.memory) if Path(args.memory).exists() else ExecutionGraphMemory()
    key_to_node_id = ensure_candidate_nodes(memory, examples, endpoints_by_url)
    add_composition_traces_to_memory(memory, train_examples, key_to_node_id)
    memory.propagate_reliability(layers=2)
    trace_scorer = TraceRagScorer(train_examples)
    scorer = ExperimentScorer(memory, trace_scorer, key_to_node_id, endpoints_by_url, args.trace_top_k)
    retriever = RoleSpecificRetriever(memory)
    return {
        "records": records,
        "examples": examples,
        "train_examples": train_examples,
        "val_examples": val_examples,
        "test_examples": test_examples,
        "endpoints_by_url": endpoints_by_url,
        "memory": memory,
        "key_to_node_id": key_to_node_id,
        "trace_scorer": trace_scorer,
        "scorer": scorer,
        "retriever": retriever,
    }


def tuned_scores(ctx: dict[str, Any], example: Any, weights: dict[str, float]) -> list[float]:
    memory = ctx["memory"]
    key_to_node_id = ctx["key_to_node_id"]
    endpoints_by_url = ctx["endpoints_by_url"]
    retriever = ctx["retriever"]
    trace_scores, _ = ctx["trace_scorer"].candidate_scores(example, top_k=16)
    node_ids = [
        key_to_node_id.get(endpoint_key(candidate)) or key_to_node_id.get(str(candidate.get("api_name") or ""))
        for candidate in example.candidates
    ]
    valid_node_ids = [node_id for node_id in node_ids if node_id in memory.nodes]
    graph_scores = retriever.score_nodes(step_text(example), "provider", valid_node_ids) if valid_node_ids else {}
    scores: list[float] = []
    for idx, candidate in enumerate(example.candidates):
        node = memory.nodes.get(node_ids[idx] or "")
        endpoint = endpoints_by_url.get(candidate.get("endpoint"))
        rank_feature = 1.0 / float(idx + 1)
        sim = float(candidate.get("similarity_score") or 0.0)
        trace = max(
            trace_scores.get(endpoint_key(candidate), 0.0),
            trace_scores.get(str(candidate.get("api_name") or ""), 0.0),
        )
        graph = float(graph_scores.get(node_ids[idx] or "", 0.0))
        reliability = float(node.reliability) if node else 0.5
        risk = float(node.risk) if node else 0.0
        observed_success = endpoint.get("observed_success") if endpoint else None
        success_hint = 0.0 if observed_success is None else (1.0 if observed_success is True else -1.0)
        score = (
            weights["rank"] * rank_feature
            + weights["sim"] * sim
            + weights["trace"] * trace
            + weights["graph"] * graph
            + weights["rel"] * reliability
            - weights["risk"] * risk
            + weights["success"] * success_hint
        )
        scores.append(score)
    return scores


def build_feature_rows(ctx: dict[str, Any], examples: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    memory = ctx["memory"]
    key_to_node_id = ctx["key_to_node_id"]
    endpoints_by_url = ctx["endpoints_by_url"]
    retriever = ctx["retriever"]
    for example in examples:
        trace_scores, _ = ctx["trace_scorer"].candidate_scores(example, top_k=16)
        node_ids = [
            key_to_node_id.get(endpoint_key(candidate)) or key_to_node_id.get(str(candidate.get("api_name") or ""))
            for candidate in example.candidates
        ]
        valid_node_ids = [node_id for node_id in node_ids if node_id in memory.nodes]
        graph_scores = retriever.score_nodes(step_text(example), "provider", valid_node_ids) if valid_node_ids else {}
        features = []
        endpoints = []
        for idx, candidate in enumerate(example.candidates):
            node = memory.nodes.get(node_ids[idx] or "")
            endpoint = endpoints_by_url.get(candidate.get("endpoint"))
            endpoints.append(endpoint)
            observed_success = endpoint.get("observed_success") if endpoint else None
            features.append(
                [
                    1.0 / float(idx + 1),
                    float(candidate.get("similarity_score") or 0.0),
                    max(
                        trace_scores.get(endpoint_key(candidate), 0.0),
                        trace_scores.get(str(candidate.get("api_name") or ""), 0.0),
                    ),
                    float(graph_scores.get(node_ids[idx] or "", 0.0)),
                    float(node.reliability) if node else 0.5,
                    float(node.risk) if node else 0.0,
                    0.0 if observed_success is None else (1.0 if observed_success is True else -1.0),
                ]
            )
        rows.append(
            {
                "record_id": example.record_id,
                "gold_index": example.gold_index,
                "features": np.asarray(features, dtype=float),
                "endpoints": endpoints,
            }
        )
    return rows


def evaluate_feature_rows(rows: list[dict[str, Any]], weights: dict[str, float]) -> dict[str, float]:
    vector = np.asarray(
        [
            weights["rank"],
            weights["sim"],
            weights["trace"],
            weights["graph"],
            weights["rel"],
            -weights["risk"],
            weights["success"],
        ],
        dtype=float,
    )
    hits = 0
    top3 = 0
    mrr = 0.0
    workflows: dict[str, list[bool]] = {}
    hazard_hits = 0
    hazard_count = 0
    for row in rows:
        scores = row["features"] @ vector
        ranked = sorted(range(len(scores)), key=lambda idx: (float(scores[idx]), -idx), reverse=True)
        correct = ranked[0] == row["gold_index"]
        hits += int(correct)
        top3 += int(row["gold_index"] in ranked[:3])
        mrr += 1.0 / (ranked.index(row["gold_index"]) + 1)
        workflows.setdefault(row["record_id"], []).append(correct)
        endpoints = row["endpoints"]
        top_failed = bool(endpoints and endpoints[0] and endpoints[0].get("observed_success") is False)
        has_success = any(endpoint and endpoint.get("observed_success") is True for endpoint in endpoints)
        if top_failed and has_success:
            hazard_count += 1
            predicted = endpoints[ranked[0]] if ranked else None
            hazard_hits += int(predicted is not None and predicted.get("observed_success") is True)
    n = len(rows)
    return {
        "api_acc": hits / n if n else 0.0,
        "api_top3": top3 / n if n else 0.0,
        "api_mrr": mrr / n if n else 0.0,
        "workflow_exact": sum(1 for values in workflows.values() if all(values)) / len(workflows) if workflows else 0.0,
        "hazard_success_proxy": hazard_hits / hazard_count if hazard_count else 0.0,
    }


def main() -> None:
    args = parse_args()
    ctx = build_context(args)
    val_rows = build_feature_rows(ctx, ctx["val_examples"])
    test_rows = build_feature_rows(ctx, ctx["test_examples"])
    weight_grid = []
    for rank_w, trace_w, graph_w, rel_w, risk_w in itertools.product(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [0.0, 0.25, 0.5],
        [0.0, 0.25],
        [0.0, 0.1, 0.25],
        [0.0, 0.1, 0.25],
    ):
        weight_grid.append(
            {
                "rank": rank_w,
                "sim": 1.0,
                "trace": trace_w,
                "graph": graph_w,
                "rel": rel_w,
                "risk": risk_w,
                "success": 0.0,
            }
        )

    rows = []
    for weights in weight_grid:
        val = evaluate_feature_rows(val_rows, weights)
        # Primary objective follows RQ1 API.Acc; hazard is secondary.
        objective = val["api_acc"] + 0.10 * val["hazard_success_proxy"]
        rows.append({"weights": weights, "val": val, "objective": objective})
    rows.sort(key=lambda row: (row["objective"], row["val"]["api_mrr"]), reverse=True)
    best = rows[0]
    best["test"] = evaluate_feature_rows(test_rows, best["weights"])

    # Also report the diagnostic transductive gate separately.
    oracle_weights = dict(best["weights"])
    oracle_weights["success"] = 0.25
    oracle = {
        "weights": oracle_weights,
        "val": evaluate_feature_rows(val_rows, oracle_weights),
        "test": evaluate_feature_rows(test_rows, oracle_weights),
        "note": "Uses observed_success at test time; diagnostic upper bound only.",
    }

    payload = {
        "config": vars(args),
        "best_validation_reranker": best,
        "oracle_success_gate": oracle,
        "top_20": rows[:20],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "oracle": oracle}, ensure_ascii=False, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
