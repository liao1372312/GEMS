#!/usr/bin/env python
"""Optimize a conservative reliability gate over semantic top-1."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_public_composition_experiments import (
    add_composition_traces_to_memory,
    endpoint_key,
    ensure_candidate_nodes,
    f1_score,
    load_composition_examples,
    load_processed_endpoint_maps,
    required_param_names,
    split_records,
)
from gems.graph_memory import ExecutionGraphMemory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--processed-endpoints", default="dataset/processed/endpoints.jsonl")
    parser.add_argument("--memory", default="outputs/gems_memory_train_only.json")
    parser.add_argument("--output", default="outputs/reliability_gate_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def build_context(args: argparse.Namespace) -> dict[str, Any]:
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    split_examples = {
        split: [example for example in examples if example.record_id in ids]
        for split, ids in splits.items()
    }
    endpoints_by_url, _ = load_processed_endpoint_maps(args.processed_endpoints)
    memory = ExecutionGraphMemory.load(args.memory) if Path(args.memory).exists() else ExecutionGraphMemory()
    key_to_node_id = ensure_candidate_nodes(memory, examples, endpoints_by_url)
    add_composition_traces_to_memory(memory, split_examples["train"], key_to_node_id)
    memory.propagate_reliability(layers=2)
    return {
        "split_examples": split_examples,
        "endpoints_by_url": endpoints_by_url,
        "memory": memory,
        "key_to_node_id": key_to_node_id,
    }


def candidate_state(ctx: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    node_id = ctx["key_to_node_id"].get(endpoint_key(candidate)) or ctx["key_to_node_id"].get(str(candidate.get("api_name") or ""))
    node = ctx["memory"].nodes.get(node_id or "")
    return {
        "reliability": float(node.reliability) if node else 0.5,
        "risk": float(node.risk) if node else 0.0,
        "score": float(candidate.get("similarity_score") or 0.0),
    }


def choose_with_gate(ctx: dict[str, Any], example: Any, config: dict[str, float]) -> int:
    top = candidate_state(ctx, example.candidates[0])
    should_gate = (
        top["risk"] >= config["top_risk_min"]
        or top["reliability"] <= config["top_rel_max"]
    )
    if not should_gate:
        return 0

    best_idx = 0
    best_value = -1e9
    top_score = top["score"]
    for idx, candidate in enumerate(example.candidates[1:], start=1):
        state = candidate_state(ctx, candidate)
        score_drop = top_score - state["score"]
        if score_drop > config["max_score_drop"]:
            continue
        if state["reliability"] < config["alt_rel_min"]:
            continue
        if state["risk"] > config["alt_risk_max"]:
            continue
        value = (
            config["rel_weight"] * state["reliability"]
            - config["risk_weight"] * state["risk"]
            - config["drop_weight"] * score_drop
            + state["score"]
        )
        if value > best_value:
            best_idx = idx
            best_value = value
    return best_idx


def evaluate(ctx: dict[str, Any], examples: list[Any], config: dict[str, float]) -> dict[str, float]:
    hits = 0
    top3 = 0
    mrr = 0.0
    workflows: dict[str, list[bool]] = {}
    para_f1 = []
    changed = 0
    hazard_hits = 0
    hazard_count = 0
    for example in examples:
        pred = choose_with_gate(ctx, example, config)
        changed += int(pred != 0)
        correct = pred == example.gold_index
        hits += int(correct)
        # Ranking diagnostic: semantic order with gated prediction moved first.
        ranking = [pred] + [idx for idx in range(len(example.candidates)) if idx != pred]
        top3 += int(example.gold_index in ranking[:3])
        mrr += 1.0 / (ranking.index(example.gold_index) + 1)
        workflows.setdefault(example.record_id, []).append(correct)
        para_f1.append(
            f1_score(
                required_param_names(example.candidates[pred]),
                required_param_names(example.gold_candidate),
            )
        )

        endpoints = [ctx["endpoints_by_url"].get(candidate.get("endpoint")) for candidate in example.candidates]
        top_failed = bool(endpoints and endpoints[0] and endpoints[0].get("observed_success") is False)
        has_success = any(endpoint and endpoint.get("observed_success") is True for endpoint in endpoints)
        if top_failed and has_success:
            hazard_count += 1
            selected = endpoints[pred]
            hazard_hits += int(selected is not None and selected.get("observed_success") is True)

    n = len(examples)
    return {
        "api_acc": hits / n if n else 0.0,
        "api_top3": top3 / n if n else 0.0,
        "api_mrr": mrr / n if n else 0.0,
        "workflow_exact": sum(1 for values in workflows.values() if all(values)) / len(workflows) if workflows else 0.0,
        "para_f1": sum(para_f1) / len(para_f1) if para_f1 else 0.0,
        "change_rate": changed / n if n else 0.0,
        "hazard_success_proxy": hazard_hits / hazard_count if hazard_count else 0.0,
    }


def main() -> None:
    args = parse_args()
    ctx = build_context(args)
    configs = []
    for top_risk_min, top_rel_max, alt_rel_min, alt_risk_max, max_score_drop in itertools.product(
        [0.10, 0.20, 0.30, 0.40, 0.50],
        [0.35, 0.45, 0.55, 0.65],
        [0.45, 0.55, 0.65, 0.75],
        [0.10, 0.20, 0.35, 0.50],
        [0.005, 0.010, 0.020, 0.040, 0.080],
    ):
        configs.append(
            {
                "top_risk_min": top_risk_min,
                "top_rel_max": top_rel_max,
                "alt_rel_min": alt_rel_min,
                "alt_risk_max": alt_risk_max,
                "max_score_drop": max_score_drop,
                "rel_weight": 0.5,
                "risk_weight": 0.5,
                "drop_weight": 2.0,
            }
        )

    rows = []
    for config in configs:
        val = evaluate(ctx, ctx["split_examples"]["val"], config)
        # Constrain accuracy degradation while improving hazard behavior.
        objective = val["api_acc"] + 0.05 * val["hazard_success_proxy"] - 0.02 * val["change_rate"]
        rows.append({"config": config, "val": val, "objective": objective})
    rows.sort(key=lambda row: (row["objective"], row["val"]["api_acc"], row["val"]["hazard_success_proxy"]), reverse=True)
    best = rows[0]
    best["test"] = evaluate(ctx, ctx["split_examples"]["test"], best["config"])

    semantic_config = {
        "top_risk_min": 999.0,
        "top_rel_max": -1.0,
        "alt_rel_min": 0.0,
        "alt_risk_max": 1.0,
        "max_score_drop": 0.0,
        "rel_weight": 0.0,
        "risk_weight": 0.0,
        "drop_weight": 0.0,
    }
    payload = {
        "best": best,
        "semantic_top1": {
            "val": evaluate(ctx, ctx["split_examples"]["val"], semantic_config),
            "test": evaluate(ctx, ctx["split_examples"]["test"], semantic_config),
        },
        "top_20": rows[:20],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"semantic_top1": payload["semantic_top1"], "best": best}, ensure_ascii=False, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
