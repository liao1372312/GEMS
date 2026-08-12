#!/usr/bin/env python
"""Evaluate a validation-selected workflow-level GEMS supervisor.

The supervisor routes each workflow to one of several GEMS operating points:
semantic top-1, conservative safe-plan verification, and more aggressive
reliability-aware LLM correction. Selection is performed on the validation
split and then frozen for the test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_gems_plan_verifier import Policy, load_llm_predictions, predict_plan, semantic_margin
from run_public_composition_experiments import (
    f1_score,
    load_composition_examples,
    required_param_names,
    split_records,
)


@dataclass(frozen=True)
class RouterConfig:
    candidate_policy: str
    fallback_policy: str
    easy_policy: str
    easy_max_steps: int
    easy_margin_max: float
    candidate_max_changes: int
    candidate_margin_max: float
    candidate_max_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--llm-files", nargs="*", default=[])
    parser.add_argument("--output", default="outputs/gems_workflow_router_eval.json")
    parser.add_argument(
        "--objective",
        choices=["balanced", "plan_guarded", "api_guarded"],
        default="balanced",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def policy_bank() -> dict[str, Policy]:
    ranks = tuple(range(2, 11))
    return {
        "semantic": Policy(1.01, 0.0, 0.0, 0.0, 0.0, 1.0, 0, 99, False, ranks),
        "safe": Policy(0.95, 0.01, 0.0, 0.05, 0.0, 0.0, 1, 2, False, ranks),
        "plan": Policy(0.9, 1.0, 0.0, 0.05, 0.002, 0.0, 2, 3, False, ranks),
        "plan_short": Policy(0.9, 1.0, 0.002, 0.05, 0.0, 0.0, 2, 2, False, ranks),
        "api": Policy(0.9, 1.0, 0.0, 0.05, 0.002, 0.0, 99, 99, False, ranks),
        "robust": Policy(0.9, 1.0, 0.0, 0.05, 0.001, 0.0, 99, 99, False, ranks),
        "margin02": Policy(0.9, 0.02, 0.0, 0.05, 0.002, 0.0, 99, 99, False, ranks),
        "all_llm": Policy(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 99, 99, False, tuple(range(1, 11))),
    }


def group_by_record(examples: list[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for example in examples:
        grouped[example.record_id].append(example)
    return {record_id: sorted(plan, key=lambda item: item.step_id) for record_id, plan in grouped.items()}


def precompute_indices(
    grouped: dict[str, list[Any]],
    predictions: dict[str, dict[str, Any]],
    policies: dict[str, Policy],
) -> dict[str, dict[str, list[int]]]:
    out: dict[str, dict[str, list[int]]] = {name: {} for name in policies}
    for record_id, plan in grouped.items():
        for name, policy in policies.items():
            out[name][record_id] = predict_plan(plan, predictions, policy)
    return out


def workflow_features(grouped: dict[str, list[Any]], precomputed: dict[str, dict[str, list[int]]]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for record_id, plan in grouped.items():
        margins = [semantic_margin(example) for example in plan]
        features[record_id] = {
            "steps": len(plan),
            "margin_max": max(margins) if margins else 1.0,
            "changes": {
                name: sum(1 for idx in policy_indices[record_id] if idx != 0)
                for name, policy_indices in precomputed.items()
            },
        }
    return features


def select_policy_for_workflow(features: dict[str, Any], config: RouterConfig) -> str:
    if (
        config.easy_policy != "none"
        and features["steps"] <= config.easy_max_steps
        and features["margin_max"] <= config.easy_margin_max
    ):
        return config.easy_policy
    if (
        features["steps"] <= config.candidate_max_steps
        and features["margin_max"] <= config.candidate_margin_max
        and features["changes"][config.candidate_policy] <= config.candidate_max_changes
    ):
        return config.candidate_policy
    return config.fallback_policy


def evaluate_config(
    grouped: dict[str, list[Any]],
    precomputed: dict[str, dict[str, list[int]]],
    features: dict[str, dict[str, Any]],
    config: RouterConfig,
) -> dict[str, Any]:
    step_hits = 0
    para_f1: list[float] = []
    workflow_hits = 0
    changed = 0
    policy_counts: dict[str, int] = defaultdict(int)
    steps = 0
    for record_id, plan in grouped.items():
        policy_name = select_policy_for_workflow(features[record_id], config)
        policy_counts[policy_name] += 1
        indices = precomputed[policy_name][record_id]
        plan_ok = True
        for example, pred_index in zip(plan, indices):
            correct = pred_index == example.gold_index
            step_hits += int(correct)
            changed += int(pred_index != 0)
            steps += 1
            plan_ok = plan_ok and correct
            para_f1.append(
                f1_score(
                    required_param_names(example.candidates[pred_index]),
                    required_param_names(example.gold_candidate),
                )
            )
        workflow_hits += int(plan_ok)
    workflows = len(grouped)
    return {
        "workflow_exact": workflow_hits / workflows if workflows else 0.0,
        "api_acc": step_hits / steps if steps else 0.0,
        "para_f1": sum(para_f1) / len(para_f1) if para_f1 else 0.0,
        "change_rate": changed / steps if steps else 0.0,
        "steps": steps,
        "workflows": workflows,
        "policy_counts": dict(sorted(policy_counts.items())),
    }


def config_grid() -> list[RouterConfig]:
    configs: list[RouterConfig] = []
    # Keep the search space intentionally small and interpretable. These axes
    # correspond to the paper-facing supervisor decisions: when to trust a
    # correction policy, how many changes a workflow may absorb, and whether
    # easy short workflows should remain conservative.
    for candidate_policy in ["safe", "plan", "plan_short", "api", "robust", "margin02"]:
        for fallback_policy in ["semantic", "safe", "plan"]:
            for easy_policy in ["none", "semantic", "safe"]:
                for candidate_max_changes in [1, 2, 3, 99]:
                    for candidate_margin_max in [0.01, 0.02, 0.05, 1.0]:
                        for candidate_max_steps in [2, 3, 99]:
                            configs.append(
                                RouterConfig(
                                    candidate_policy=candidate_policy,
                                    fallback_policy=fallback_policy,
                                    easy_policy=easy_policy,
                                    easy_max_steps=2,
                                    easy_margin_max=0.01,
                                    candidate_max_changes=candidate_max_changes,
                                    candidate_margin_max=candidate_margin_max,
                                    candidate_max_steps=candidate_max_steps,
                                )
                            )
    return configs


def policy_metrics(
    grouped: dict[str, list[Any]],
    precomputed: dict[str, dict[str, list[int]]],
    policy_name: str,
) -> dict[str, Any]:
    config = RouterConfig(policy_name, policy_name, "none", 0, 0.0, 99, 1.0, 99)
    features = workflow_features(grouped, precomputed)
    return evaluate_config(grouped, precomputed, features, config)


def objective_tuple(metrics: dict[str, Any], semantic_metrics: dict[str, Any], objective: str) -> tuple[Any, ...]:
    balanced = 0.34 * metrics["workflow_exact"] + 0.33 * metrics["api_acc"] + 0.33 * metrics["para_f1"]
    if objective == "plan_guarded":
        return (
            metrics["workflow_exact"] >= semantic_metrics["workflow_exact"] - 0.005,
            balanced,
            metrics["workflow_exact"],
            metrics["api_acc"],
            metrics["para_f1"],
            -metrics["change_rate"],
        )
    if objective == "api_guarded":
        return (
            metrics["workflow_exact"] >= semantic_metrics["workflow_exact"] - 0.03,
            metrics["api_acc"],
            metrics["para_f1"],
            metrics["workflow_exact"],
            -metrics["change_rate"],
        )
    return (
        balanced,
        metrics["workflow_exact"],
        metrics["api_acc"],
        metrics["para_f1"],
        -metrics["change_rate"],
    )


def main() -> None:
    args = parse_args()
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    split_examples = {
        split: [example for example in examples if example.record_id in ids]
        for split, ids in splits.items()
    }
    predictions = load_llm_predictions(args.llm_files)
    policies = policy_bank()
    grouped = {split: group_by_record(items) for split, items in split_examples.items()}
    precomputed = {
        split: precompute_indices(items, predictions, policies)
        for split, items in grouped.items()
    }
    features = {
        split: workflow_features(items, precomputed[split])
        for split, items in grouped.items()
    }

    semantic_val = policy_metrics(grouped["val"], precomputed["val"], "semantic")
    rows = []
    for config in config_grid():
        metrics = evaluate_config(grouped["val"], precomputed["val"], features["val"], config)
        rows.append(
            {
                "config": config.__dict__,
                "metrics": metrics,
                "objective": objective_tuple(metrics, semantic_val, args.objective),
            }
        )
    rows.sort(key=lambda row: row["objective"], reverse=True)
    best_config = RouterConfig(**rows[0]["config"])
    test_metrics = evaluate_config(grouped["test"], precomputed["test"], features["test"], best_config)

    payload = {
        "paper_ready": True,
        "selection_split": "val",
        "objective": args.objective,
        "best_config": best_config.__dict__,
        "val": rows[0]["metrics"],
        "test": test_metrics,
        "single_policy_val": {
            name: policy_metrics(grouped["val"], precomputed["val"], name)
            for name in policies
        },
        "single_policy_test": {
            name: policy_metrics(grouped["test"], precomputed["test"], name)
            for name in policies
        },
        "top_20": rows[:20],
        "available_llm_predictions": len(predictions),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
