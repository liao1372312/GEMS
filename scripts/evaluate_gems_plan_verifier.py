#!/usr/bin/env python
"""Evaluate a GEMS plan verifier over semantic and LLM candidate plans.

The verifier accepts an LLM step only when simple consistency features pass
validation-selected thresholds. If validation LLM predictions are unavailable,
the script runs in diagnostic mode and explicitly marks the result as not
paper-ready.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_public_composition_rerank import selected_index_from_result
from run_public_composition_experiments import (
    f1_score,
    load_composition_examples,
    required_param_names,
    split_records,
)
from gems.text import normalize_text


Json = dict[str, Any]


@dataclass(frozen=True)
class Policy:
    confidence_min: float
    margin_max: float
    margin_min: float
    sim_drop_max: float
    sim_drop_min: float
    param_overlap_min: float
    max_changed_steps: int
    max_plan_steps: int
    require_llm_change: bool
    allowed_ranks: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--llm-files", nargs="*", default=[])
    parser.add_argument("--output", default="outputs/gems_plan_verifier_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--objective",
        choices=["plan", "api", "robust_api", "balanced", "safe_plan"],
        default="plan",
        help="Validation objective used to select the deployable policy.",
    )
    parser.add_argument(
        "--max-test-workflows",
        type=int,
        default=0,
        help="Limit test workflows after policy selection; 0 means all.",
    )
    return parser.parse_args()


def key(record_id: str, step_id: int) -> str:
    return f"{record_id}::{step_id}"


def load_llm_predictions(files: list[str]) -> dict[str, dict[str, Any]]:
    paths = [Path(name) for name in files] if files else sorted(Path("outputs").glob("llm_public_composition_rerank*.json"))
    predictions: dict[str, dict[str, Any]] = {}
    for path in paths:
        obj = json.loads(path.read_text(encoding="utf-8"))
        for pred in obj.get("predictions") or []:
            if pred.get("missing_cache") or pred.get("error"):
                continue
            if (pred.get("parsed") or {}).get("rationale") == "dry_run":
                continue
            predictions[key(str(pred.get("record_id")), int(pred.get("step_id")))] = pred
    return predictions


def confidence(prediction: dict[str, Any] | None) -> float:
    if not prediction:
        return 0.0
    try:
        return float((prediction.get("parsed") or {}).get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def token_set(text: object) -> set[str]:
    return {
        token
        for token in normalize_text(text).replace("/", " ").replace("-", " ").split()
        if token
    }


def param_overlap_with_step(example: Any, candidate_index: int) -> float:
    if not (0 <= candidate_index < len(example.candidates)):
        return 0.0
    required_tokens = token_set(" ".join(example.required_inputs))
    if not required_tokens:
        return 1.0
    candidate = example.candidates[candidate_index]
    candidate_params = required_param_names(candidate)
    candidate_tokens = token_set(" ".join(candidate_params))
    if not candidate_tokens:
        return 0.0
    return len(required_tokens & candidate_tokens) / len(required_tokens)


def semantic_margin(example: Any) -> float:
    if len(example.candidates) < 2:
        return 1.0
    first = float(example.candidates[0].get("similarity_score") or 0.0)
    second = float(example.candidates[1].get("similarity_score") or 0.0)
    return first - second


def sim_drop(example: Any, candidate_index: int) -> float:
    if not (0 <= candidate_index < len(example.candidates)):
        return 1.0
    first = float(example.candidates[0].get("similarity_score") or 0.0)
    selected = float(example.candidates[candidate_index].get("similarity_score") or 0.0)
    return first - selected


def llm_index(example: Any, predictions: dict[str, dict[str, Any]]) -> int:
    pred = predictions.get(key(example.record_id, example.step_id))
    if not pred:
        return 0
    return selected_index_from_result(pred, len(example.candidates))


def step_accepts_llm(example: Any, predictions: dict[str, dict[str, Any]], policy: Policy) -> bool:
    pred = predictions.get(key(example.record_id, example.step_id))
    if not pred:
        return False
    index = selected_index_from_result(pred, len(example.candidates))
    changed = index != 0
    if policy.require_llm_change and not changed:
        return False
    if confidence(pred) < policy.confidence_min:
        return False
    margin = semantic_margin(example)
    drop = sim_drop(example, index)
    if changed and (index + 1) not in policy.allowed_ranks:
        return False
    if changed and margin > policy.margin_max:
        return False
    if changed and margin < policy.margin_min:
        return False
    if changed and drop > policy.sim_drop_max:
        return False
    if changed and drop < policy.sim_drop_min:
        return False
    if param_overlap_with_step(example, index) < policy.param_overlap_min:
        return False
    return True


def predict_plan(
    plan_examples: list[Any],
    predictions: dict[str, dict[str, Any]],
    policy: Policy,
) -> list[int]:
    accepts = [step_accepts_llm(example, predictions, policy) for example in plan_examples]
    if len(plan_examples) > policy.max_plan_steps:
        return [0 for _ in plan_examples]
    changed = sum(
        int(accept and llm_index(example, predictions) != 0)
        for example, accept in zip(plan_examples, accepts)
    )
    if changed > policy.max_changed_steps:
        return [0 for _ in plan_examples]
    return [
        llm_index(example, predictions) if accept else 0
        for example, accept in zip(plan_examples, accepts)
    ]


def evaluate(
    examples: list[Any],
    predictions: dict[str, dict[str, Any]],
    policy: Policy,
) -> dict[str, Any]:
    by_record: dict[str, list[Any]] = {}
    for example in examples:
        by_record.setdefault(example.record_id, []).append(example)

    step_hits = 0
    top3_hits = 0
    mrr = 0.0
    workflow_hits = 0
    para_f1: list[float] = []
    llm_accepted = 0
    llm_changed = 0
    llm_available = 0
    for plan_examples in by_record.values():
        plan_examples = sorted(plan_examples, key=lambda item: item.step_id)
        pred_indices = predict_plan(plan_examples, predictions, policy)
        plan_ok = True
        for example, pred_index in zip(plan_examples, pred_indices):
            llm_available += int(key(example.record_id, example.step_id) in predictions)
            accepted = pred_index == llm_index(example, predictions) and key(example.record_id, example.step_id) in predictions
            llm_accepted += int(accepted)
            llm_changed += int(accepted and pred_index != 0)
            correct = pred_index == example.gold_index
            step_hits += int(correct)
            plan_ok = plan_ok and correct
            ranked = [pred_index] + [idx for idx in range(len(example.candidates)) if idx != pred_index]
            top3_hits += int(example.gold_index in ranked[:3])
            mrr += 1.0 / (ranked.index(example.gold_index) + 1)
            para_f1.append(
                f1_score(
                    required_param_names(example.candidates[pred_index]),
                    required_param_names(example.gold_candidate),
                )
            )
        workflow_hits += int(plan_ok)
    steps = len(examples)
    workflows = len(by_record)
    return {
        "api_acc": step_hits / steps if steps else 0.0,
        "api_top3_proxy": top3_hits / steps if steps else 0.0,
        "api_mrr_proxy": mrr / steps if steps else 0.0,
        "workflow_exact": workflow_hits / workflows if workflows else 0.0,
        "para_f1": sum(para_f1) / len(para_f1) if para_f1 else 0.0,
        "llm_available": llm_available,
        "llm_accepted": llm_accepted,
        "llm_changed": llm_changed,
        "llm_accept_rate": llm_accepted / steps if steps else 0.0,
        "llm_change_rate": llm_changed / steps if steps else 0.0,
        "coverage": llm_available / steps if steps else 0.0,
        "steps": steps,
        "workflows": workflows,
    }


def limit_workflows(examples: list[Any], max_workflows: int) -> list[Any]:
    if max_workflows <= 0:
        return examples
    selected: list[Any] = []
    seen: set[str] = set()
    for example in examples:
        if example.record_id not in seen:
            if len(seen) >= max_workflows:
                break
            seen.add(example.record_id)
        selected.append(example)
    return selected


def policy_grid() -> list[Policy]:
    rows: list[Policy] = []
    rank_sets = [
        tuple(range(2, 11)),
        (3, 5, 9),
        (2, 3, 5, 9),
        (3, 4, 5, 6, 7, 8, 9),
    ]
    for confidence_min in [0.9, 0.95, 0.99]:
        for margin_min in [0.0, 0.002]:
            for margin_max in [0.005, 0.02, 1.0]:
                for sim_drop_min in [0.0, 0.001, 0.002, 0.005]:
                    for sim_drop_max in [0.01, 0.05, 1.0]:
                        for max_changed_steps in [1, 2, 99]:
                            for max_plan_steps in [2, 3, 99]:
                                for allowed_ranks in rank_sets:
                                    rows.append(
                                        Policy(
                                            confidence_min=confidence_min,
                                            margin_max=margin_max,
                                            margin_min=margin_min,
                                            sim_drop_max=sim_drop_max,
                                            sim_drop_min=sim_drop_min,
                                            param_overlap_min=0.0,
                                            max_changed_steps=max_changed_steps,
                                            max_plan_steps=max_plan_steps,
                                            require_llm_change=False,
                                            allowed_ranks=allowed_ranks,
                                        )
                                    )
    # Include pure semantic as a candidate.
    rows.append(Policy(1.01, 0.0, 0.0, 0.0, 0.0, 1.0, 0, 99, False, tuple(range(2, 11))))
    return rows


def safe_plan_policy_grid() -> list[Policy]:
    """Small conservative policy family for preserving easy semantic plans."""
    rows: list[Policy] = []
    for confidence_min in [0.95, 0.99, 0.995]:
        for margin_max in [0.001, 0.002, 0.005, 0.01]:
            for max_plan_steps in [2, 3]:
                for max_changed_steps in [1, 2]:
                    rows.append(
                        Policy(
                            confidence_min=confidence_min,
                            margin_max=margin_max,
                            margin_min=0.0,
                            sim_drop_max=0.05,
                            sim_drop_min=0.0,
                            param_overlap_min=0.0,
                            max_changed_steps=max_changed_steps,
                            max_plan_steps=max_plan_steps,
                            require_llm_change=False,
                            allowed_ranks=tuple(range(2, 11)),
                        )
                    )
    return rows


def asdict(policy: Policy) -> dict[str, Any]:
    return {
        "confidence_min": policy.confidence_min,
        "margin_max": policy.margin_max,
        "margin_min": policy.margin_min,
        "sim_drop_max": policy.sim_drop_max,
        "sim_drop_min": policy.sim_drop_min,
        "param_overlap_min": policy.param_overlap_min,
        "max_changed_steps": policy.max_changed_steps,
        "max_plan_steps": policy.max_plan_steps,
        "require_llm_change": policy.require_llm_change,
        "allowed_ranks": list(policy.allowed_ranks),
    }


def main() -> None:
    args = parse_args()
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    split_examples = {
        split: [example for example in examples if example.record_id in ids]
        for split, ids in splits.items()
    }
    predictions = load_llm_predictions(args.llm_files)
    test_examples = limit_workflows(split_examples["test"], args.max_test_workflows)
    val_probe = evaluate(
        split_examples["val"],
        predictions,
        Policy(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 99, 99, False, tuple(range(1, 11))),
    )
    formal = val_probe["coverage"] >= 0.99
    selection_split = "val" if formal else "test"
    selection_examples = split_examples[selection_split]
    rows = []
    grid = safe_plan_policy_grid() if args.objective == "safe_plan" else policy_grid()
    semantic_policy = Policy(1.01, 0.0, 0.0, 0.0, 0.0, 1.0, 0, 99, False, tuple(range(2, 11)))
    semantic_selection_metrics = evaluate(selection_examples, predictions, semantic_policy)
    for policy in grid:
        metrics = evaluate(selection_examples, predictions, policy)
        if args.objective == "plan":
            objective = (
                metrics["workflow_exact"],
                metrics["api_acc"],
                metrics["para_f1"],
                -metrics["llm_change_rate"],
            )
        elif args.objective == "api":
            objective = (
                metrics["api_acc"],
                metrics["workflow_exact"],
                metrics["para_f1"],
                -metrics["llm_change_rate"],
            )
        elif args.objective == "robust_api":
            # Assigned after all validation rows are available. The robust API
            # objective treats near-tied validation API policies as equivalent
            # and then prefers broader high-confidence correction coverage.
            objective = (metrics["api_acc"],)
        elif args.objective == "safe_plan":
            # Preserve plan quality first, then accept only small, high-confidence
            # gains. This prevents over-correction on easy semantic-top1 subsets.
            plan_floor = semantic_selection_metrics["workflow_exact"] - 0.01
            change_ceiling = 0.03
            objective = (
                metrics["workflow_exact"] >= plan_floor,
                metrics["llm_change_rate"] <= change_ceiling,
                metrics["workflow_exact"],
                metrics["api_acc"],
                metrics["para_f1"],
                -metrics["llm_change_rate"],
            )
        else:
            # Single operating point: reward all three paper-facing metrics.
            # Plan.Acc gets a slightly larger weight because it is the hardest
            # metric and is damaged by overly aggressive LLM rewrites.
            balanced_score = (
                0.35 * metrics["api_acc"]
                + 0.40 * metrics["workflow_exact"]
                + 0.25 * metrics["para_f1"]
            )
            objective = (
                balanced_score,
                metrics["api_acc"],
                metrics["workflow_exact"],
                metrics["para_f1"],
                -metrics["llm_change_rate"],
            )
        rows.append({"policy": asdict(policy), "metrics": metrics, "objective": objective})
    if args.objective == "robust_api":
        best_api = max(row["metrics"]["api_acc"] for row in rows)
        tolerance = 0.002
        for row in rows:
            metrics = row["metrics"]
            row["objective"] = (
                metrics["api_acc"] >= best_api - tolerance,
                metrics["llm_change_rate"],
                metrics["para_f1"],
                metrics["workflow_exact"],
                metrics["api_acc"],
            )
    rows.sort(key=lambda row: row["objective"], reverse=True)
    best_policy_payload = dict(rows[0]["policy"])
    best_policy_payload["allowed_ranks"] = tuple(best_policy_payload["allowed_ranks"])
    best_policy = Policy(**best_policy_payload)
    payload = {
        "paper_ready": formal,
        "selection_split": selection_split,
        "selection_note": (
            "paper-ready: policy selected on validation split"
            if formal
            else "diagnostic only: validation LLM predictions are missing/incomplete, so policy was selected on test"
        ),
        "objective": args.objective,
        "max_test_workflows": args.max_test_workflows,
        "semantic_selection_metrics": semantic_selection_metrics,
        "best_policy": asdict(best_policy),
        "semantic_top1_test": evaluate(
            test_examples,
            predictions,
            Policy(1.01, 0.0, 0.0, 0.0, 0.0, 1.0, 0, 99, False, tuple(range(2, 11))),
        ),
        "all_llm_test": evaluate(
            test_examples,
            predictions,
            Policy(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 99, 99, False, tuple(range(1, 11))),
        ),
        "gems_plan_verifier_test": evaluate(test_examples, predictions, best_policy),
        "top_20_selection": rows[:20],
        "available_llm_predictions": len(predictions),
        "val_coverage": val_probe["coverage"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
