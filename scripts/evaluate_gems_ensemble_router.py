#!/usr/bin/env python
"""Evaluate validation-selected GEMS ensemble routers from cached predictions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_gems_plan_verifier import Policy, confidence, key, llm_index, predict_plan, semantic_margin
from llm_public_composition_rerank import selected_index_from_result
from run_public_composition_experiments import f1_score, load_composition_examples, required_param_names, split_records


@dataclass(frozen=True)
class RouterConfig:
    name: str
    plan_steps_max: int
    margin_max: float
    direct_conf_min: float
    rest_conf_min: float
    fallback: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--llm-baselines", default="outputs/llm_public_composition_baselines_test_all.json")
    parser.add_argument("--output", default="outputs/gems_ensemble_router_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def load_baseline_predictions(path: str) -> dict[str, dict[str, dict[str, Any]]]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for name, result in (obj.get("results") or {}).items():
        out[name] = {
            key(str(pred.get("record_id")), int(pred.get("step_id"))): pred
            for pred in result.get("predictions") or []
            if not pred.get("missing_cache") and not pred.get("error")
        }
    return out


SAFE_POLICY = Policy(0.95, 0.01, 0.0, 0.05, 0.0, 0.0, 1, 2, False, tuple(range(2, 11)))
ROBUST_POLICY = Policy(0.9, 1.0, 0.0, 0.05, 0.001, 0.0, 99, 99, False, tuple(range(2, 11)))
PLAN_POLICY = Policy(0.9, 1.0, 0.0, 0.05, 0.002, 0.0, 2, 3, False, tuple(range(2, 11)))


def group_by_record(examples: list[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for example in examples:
        grouped[example.record_id].append(example)
    return {record_id: sorted(plan, key=lambda item: item.step_id) for record_id, plan in grouped.items()}


def baseline_index(example: Any, predictions: dict[str, dict[str, dict[str, Any]]], name: str) -> int:
    pred = predictions.get(name, {}).get(key(example.record_id, example.step_id))
    if not pred:
        return 0
    return selected_index_from_result(pred, len(example.candidates))


def baseline_conf(example: Any, predictions: dict[str, dict[str, dict[str, Any]]], name: str) -> float:
    pred = predictions.get(name, {}).get(key(example.record_id, example.step_id))
    if not pred:
        return 0.0
    return confidence(pred)


def policy_indices(plan: list[Any], rerank_predictions: dict[str, dict[str, Any]], policy: Policy) -> dict[tuple[str, int], int]:
    pred_indices = predict_plan(plan, rerank_predictions, policy)
    return {(example.record_id, example.step_id): idx for example, idx in zip(plan, pred_indices)}


def select_indices_for_plan(
    plan: list[Any],
    rerank_predictions: dict[str, dict[str, Any]],
    baseline_predictions: dict[str, dict[str, dict[str, Any]]],
    config: RouterConfig,
) -> dict[tuple[str, int], int]:
    safe = policy_indices(plan, rerank_predictions, SAFE_POLICY)
    robust = policy_indices(plan, rerank_predictions, ROBUST_POLICY)
    plan_policy = policy_indices(plan, rerank_predictions, PLAN_POLICY)
    plan_steps = len(plan)
    max_margin = max((semantic_margin(example) for example in plan), default=1.0)

    chosen: dict[tuple[str, int], int] = {}
    for example in plan:
        step_key = (example.record_id, example.step_id)
        semantic_idx = 0
        safe_idx = safe[step_key]
        robust_idx = robust[step_key]
        plan_idx = plan_policy[step_key]
        direct_idx = baseline_index(example, baseline_predictions, "direct_llm")
        rest_idx = baseline_index(example, baseline_predictions, "restgpt")
        direct_conf = baseline_conf(example, baseline_predictions, "direct_llm")
        rest_conf = baseline_conf(example, baseline_predictions, "restgpt")

        if plan_steps <= config.plan_steps_max and max_margin <= config.margin_max:
            idx = safe_idx
        elif rest_idx != 0 and rest_conf >= config.rest_conf_min:
            idx = rest_idx
        elif direct_idx != 0 and direct_conf >= config.direct_conf_min:
            idx = direct_idx
        elif config.fallback == "robust":
            idx = robust_idx
        elif config.fallback == "plan":
            idx = plan_idx
        elif config.fallback == "direct":
            idx = direct_idx
        elif config.fallback == "restgpt":
            idx = rest_idx
        else:
            idx = semantic_idx
        chosen[step_key] = idx
    return chosen


def evaluate_chosen(examples: list[Any], chosen: dict[tuple[str, int], int]) -> dict[str, Any]:
    hits = 0
    workflows: dict[str, list[bool]] = defaultdict(list)
    para_values: list[float] = []
    changed = 0
    for example in examples:
        idx = chosen.get((example.record_id, example.step_id), 0)
        correct = idx == example.gold_index
        hits += int(correct)
        changed += int(idx != 0)
        workflows[example.record_id].append(correct)
        para_values.append(
            f1_score(
                required_param_names(example.candidates[idx]),
                required_param_names(example.gold_candidate),
            )
        )
    steps = len(examples)
    return {
        "api_acc": hits / steps if steps else 0.0,
        "workflow_exact": sum(1 for values in workflows.values() if values and all(values)) / len(workflows) if workflows else 0.0,
        "para_f1": sum(para_values) / len(para_values) if para_values else 0.0,
        "change_rate": changed / steps if steps else 0.0,
        "steps": steps,
        "workflows": len(workflows),
    }


def evaluate_router(
    examples: list[Any],
    rerank_predictions: dict[str, dict[str, Any]],
    baseline_predictions: dict[str, dict[str, dict[str, Any]]],
    config: RouterConfig,
) -> dict[str, Any]:
    chosen: dict[tuple[str, int], int] = {}
    for plan in group_by_record(examples).values():
        chosen.update(select_indices_for_plan(plan, rerank_predictions, baseline_predictions, config))
    return evaluate_chosen(examples, chosen)


def rerank_predictions_from_outputs() -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for path in sorted(Path("outputs").glob("llm_public_composition_rerank*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        for pred in obj.get("predictions") or []:
            if pred.get("missing_cache") or pred.get("error"):
                continue
            if (pred.get("parsed") or {}).get("rationale") == "dry_run":
                continue
            predictions[key(str(pred.get("record_id")), int(pred.get("step_id")))] = pred
    return predictions


def main() -> None:
    args = parse_args()
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    split_examples = {split: [example for example in examples if example.record_id in ids] for split, ids in splits.items()}
    rerank_predictions = rerank_predictions_from_outputs()
    baseline_predictions = load_baseline_predictions(args.llm_baselines)

    configs: list[RouterConfig] = []
    for plan_steps_max in [1, 2, 3, 99]:
        for margin_max in [0.002, 0.005, 0.01, 0.02, 1.0]:
            for direct_conf_min in [0.9, 0.95, 0.99, 1.01]:
                for rest_conf_min in [0.9, 0.95, 0.99, 1.01]:
                    for fallback in ["semantic", "safe", "plan", "robust", "direct", "restgpt"]:
                        configs.append(
                            RouterConfig(
                                name=f"ps{plan_steps_max}_m{margin_max}_dc{direct_conf_min}_rc{rest_conf_min}_{fallback}",
                                plan_steps_max=plan_steps_max,
                                margin_max=margin_max,
                                direct_conf_min=direct_conf_min,
                                rest_conf_min=rest_conf_min,
                                fallback=fallback,
                            )
                        )

    val_rows = []
    for config in configs:
        metrics = evaluate_router(split_examples["val"], rerank_predictions, baseline_predictions, config)
        balanced = 0.34 * metrics["workflow_exact"] + 0.33 * metrics["api_acc"] + 0.33 * metrics["para_f1"]
        objective = (
            balanced,
            metrics["workflow_exact"],
            metrics["api_acc"],
            metrics["para_f1"],
            -metrics["change_rate"],
        )
        val_rows.append({"config": config.__dict__, "metrics": metrics, "objective": objective})
    val_rows.sort(key=lambda row: row["objective"], reverse=True)
    best_config = RouterConfig(**val_rows[0]["config"])
    test_metrics = evaluate_router(split_examples["test"], rerank_predictions, baseline_predictions, best_config)

    payload = {
        "paper_ready": True,
        "selection_split": "val",
        "best_config": best_config.__dict__,
        "val": val_rows[0]["metrics"],
        "test": test_metrics,
        "top_20": val_rows[:20],
        "available_rerank_predictions": len(rerank_predictions),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
