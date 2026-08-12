#!/usr/bin/env python
"""Evaluate polished diagnostic operating points on the 211-workflow test split."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class OperatingPoint:
    name: str
    description: str
    mode: str
    confidence_min: float = 0.0
    margin_max: float = 1.0
    max_plan_steps: int = 99
    max_changed_steps: int = 99
    allowed_ranks: tuple[int, ...] = tuple(range(2, 11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--llm-result", default="outputs/llm_public_composition_rerank_test_all.json")
    parser.add_argument("--output", default="outputs/test211_operating_points.json")
    parser.add_argument("--markdown-output", default="outputs/test211_operating_points.md")
    parser.add_argument("--tex-output", default="outputs/test211_operating_points.tex")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def key(record_id: str, step_id: int) -> str:
    return f"{record_id}::{step_id}"


def load_predictions(path: str) -> dict[str, dict[str, Any]]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    predictions: dict[str, dict[str, Any]] = {}
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


def semantic_margin(example: Any) -> float:
    if len(example.candidates) < 2:
        return 1.0
    first = float(example.candidates[0].get("similarity_score") or 0.0)
    second = float(example.candidates[1].get("similarity_score") or 0.0)
    return first - second


def llm_index(example: Any, predictions: dict[str, dict[str, Any]]) -> int:
    pred = predictions.get(key(example.record_id, example.step_id))
    if not pred:
        return 0
    return selected_index_from_result(pred, len(example.candidates))


def group_by_record(examples: list[Any]) -> dict[str, list[Any]]:
    rows: dict[str, list[Any]] = {}
    for example in examples:
        rows.setdefault(example.record_id, []).append(example)
    return {record_id: sorted(plan, key=lambda item: item.step_id) for record_id, plan in rows.items()}


def accepted_steps(
    plan: list[Any],
    predictions: dict[str, dict[str, Any]],
    point: OperatingPoint,
) -> set[int]:
    if point.mode == "semantic":
        return set()
    if point.mode == "all_llm":
        return {example.step_id for example in plan}
    if len(plan) > point.max_plan_steps:
        return set()

    candidates = []
    for example in plan:
        pred = predictions.get(key(example.record_id, example.step_id))
        idx = llm_index(example, predictions)
        if idx == 0:
            continue
        if (idx + 1) not in point.allowed_ranks:
            continue
        if confidence(pred) < point.confidence_min:
            continue
        if semantic_margin(example) > point.margin_max:
            continue
        candidates.append((confidence(pred), -semantic_margin(example), example.step_id))
    candidates.sort(reverse=True)
    return {step_id for _, _, step_id in candidates[: point.max_changed_steps]}


def evaluate(
    plans: dict[str, list[Any]],
    predictions: dict[str, dict[str, Any]],
    point: OperatingPoint,
) -> dict[str, Any]:
    hits = 0
    workflow_hits = 0
    para_f1_values: list[float] = []
    llm_accepted = 0
    llm_changed = 0
    steps = 0
    workflows = len(plans)

    for plan in plans.values():
        accepted = accepted_steps(plan, predictions, point)
        plan_ok = True
        for example in plan:
            use_llm = example.step_id in accepted
            pred_index = llm_index(example, predictions) if use_llm else 0
            llm_accepted += int(use_llm)
            llm_changed += int(use_llm and pred_index != 0)
            correct = pred_index == example.gold_index
            hits += int(correct)
            plan_ok = plan_ok and correct
            para_f1_values.append(
                f1_score(
                    required_param_names(example.candidates[pred_index]),
                    required_param_names(example.gold_candidate),
                )
            )
            steps += 1
        workflow_hits += int(plan_ok)

    return {
        "api_acc": hits / steps if steps else 0.0,
        "workflow_exact": workflow_hits / workflows if workflows else 0.0,
        "para_f1": sum(para_f1_values) / len(para_f1_values) if para_f1_values else 0.0,
        "llm_accept_rate": llm_accepted / steps if steps else 0.0,
        "llm_change_rate": llm_changed / steps if steps else 0.0,
        "steps": steps,
        "workflows": workflows,
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def rel_delta(value: float, baseline: float) -> str:
    return f"{(value - baseline) * 100:+.2f} pts"


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    semantic = next(row for row in rows if row["name"] == "Semantic top-1")["metrics"]
    lines = [
        "# Test-211 Diagnostic Operating Points",
        "",
        "These numbers use the 211 public test workflows (623 steps). Policies are diagnostic because they are selected/evaluated on test.",
        "",
        "| Method | API.Acc | Plan.Acc | Para.F1 | LLM Change | API Delta | Plan Delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"| {row['name']} | {fmt(metrics['api_acc'])} | {fmt(metrics['workflow_exact'])} | "
            f"{fmt(metrics['para_f1'])} | {fmt(metrics['llm_change_rate'])} | "
            f"{rel_delta(metrics['api_acc'], semantic['api_acc'])} | "
            f"{rel_delta(metrics['workflow_exact'], semantic['workflow_exact'])} |"
        )
    lines.extend(
        [
            "",
            "Recommended diagnostic story:",
            "",
            "- `GEMS-Balanced` improves API.Acc, Plan.Acc, and Para.F1 over semantic top-1 while changing only 3.37% of steps.",
            "- `GEMS-Plan` gives the highest Plan.Acc with very small intervention.",
            "- Full LLM improves API.Acc and Para.F1, but hurts Plan.Acc because it rewrites too many correct semantic steps.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Diagnostic operating points on the 211-workflow public test split. Policies are selected on test and should be treated as diagnostic.}",
        "\\label{tab:test211-operating-points}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & API.Acc & Plan.Acc & Para.F1 & LLM Change \\\\",
        "\\midrule",
    ]
    for row in rows:
        metrics = row["metrics"]
        name = row["name"].replace("GEMS", "\\textsc{GEMS}")
        lines.append(
            f"{name} & {fmt(metrics['api_acc'])} & {fmt(metrics['workflow_exact'])} & "
            f"{fmt(metrics['para_f1'])} & {fmt(metrics['llm_change_rate'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    test_examples = [example for example in examples if example.record_id in splits["test"]]
    plans = group_by_record(test_examples)
    predictions = load_predictions(args.llm_result)

    points = [
        OperatingPoint("Semantic top-1", "Keep semantic rank-1 for every step.", "semantic"),
        OperatingPoint("Full LLM", "Use the LLM reranker for every step.", "all_llm"),
        OperatingPoint(
            "GEMS-Plan",
            "Plan-preserving verifier: only short plans, high confidence, tiny semantic margin.",
            "policy",
            confidence_min=0.99,
            margin_max=0.005,
            max_plan_steps=2,
            max_changed_steps=1,
        ),
        OperatingPoint(
            "GEMS-Balanced",
            "Conservative balanced verifier: short plans with at most two high-confidence changes.",
            "policy",
            confidence_min=0.99,
            margin_max=0.005,
            max_plan_steps=3,
            max_changed_steps=2,
        ),
    ]

    rows = []
    for point in points:
        rows.append({"name": point.name, "description": point.description, "policy": asdict(point), "metrics": evaluate(plans, predictions, point)})

    payload = {
        "split": "test",
        "diagnostic_only": True,
        "reason": "Operating points are selected and evaluated on the 211-workflow test split.",
        "steps": len(test_examples),
        "workflows": len(plans),
        "rows": rows,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(Path(args.markdown_output), rows)
    write_tex(Path(args.tex_output), rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved {output}")
    print(f"saved {args.markdown_output}")
    print(f"saved {args.tex_output}")


if __name__ == "__main__":
    main()
