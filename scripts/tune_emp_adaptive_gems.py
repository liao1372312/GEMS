#!/usr/bin/env python
"""Tune an adaptive GEMS scorer for the EMP composition benchmark.

The weight search uses only the validation split. Test metrics are computed
once with the selected weights.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_emp_composition_exec import (  # noqa: E402
    EmpApi,
    EmpScorer,
    EmpStep,
    load_emp_quality,
    load_records,
    param_f1,
    simulated_step_success,
    split_records,
)


Json = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/emp_data2_synthetic_compositions.jsonl")
    parser.add_argument("--emp-xlsx", default="dataset/EMP/data2.xlsx")
    parser.add_argument("--output", default="outputs/emp_adaptive_gems_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--trace-top-k", type=int, default=16)
    return parser.parse_args()


FEATURES = [
    "rank",
    "sim",
    "trace",
    "service",
    "operation",
    "param",
    "quality",
]


def candidate_features(scorer: EmpScorer, step: EmpStep, top_k: int) -> list[tuple[EmpApi, dict[str, float]]]:
    candidates = scorer.semantic_candidates(step, top_k)
    traces = scorer.trace_scores(step)
    rows: list[tuple[EmpApi, dict[str, float]]] = []
    for index, (api, sim) in enumerate(candidates):
        rows.append(
            (
                api,
                {
                    "rank": 1.0 / float(index + 1),
                    "sim": float(sim),
                    "trace": float(traces.get(api.record_id, 0.0)),
                    "service": 1.0 if api.service_cn and api.service_cn in step.step_description else 0.0,
                    "operation": 1.0 if api.operation_type and api.operation_type in step.step_description else 0.0,
                    "param": float(scorer.param_overlap(step, api)),
                    "quality": float(api.quality),
                },
            )
        )
    return rows


def build_rows(scorer: EmpScorer, steps: list[EmpStep], top_k: int) -> list[dict[str, Any]]:
    rows = []
    step_index: dict[tuple[str, int], EmpStep] = {(step.record_id, step.step_id): step for step in steps}
    for step in steps:
        candidates = candidate_features(scorer, step, top_k)
        prior_steps = {
            previous_id
            for (record_id, previous_id), previous in step_index.items()
            if record_id == step.record_id and previous.step_id < step.step_id
        }
        rows.append(
            {
                "record_id": step.record_id,
                "step": step,
                "gold_api_id": step.gold_api_id,
                "dependency_ok": all(dep in prior_steps for dep in step.depends_on),
                "candidates": [
                    {
                        "api": api,
                        "features": features,
                    }
                    for api, features in candidates
                ],
            }
        )
    return rows


def predict_weighted(
    row: dict[str, Any],
    weights: dict[str, float],
) -> tuple[EmpApi, list[EmpApi]]:
    scored = []
    for item in row["candidates"]:
        api = item["api"]
        features = item["features"]
        score = sum(weights[name] * features[name] for name in FEATURES)
        scored.append((float(score), api))
    scored.sort(key=lambda item: (item[0], item[1].record_id), reverse=True)
    return scored[0][1], [api for _, api in scored]


def evaluate(
    rows: list[dict[str, Any]],
    weights: dict[str, float],
) -> dict[str, Any]:
    hits = 0
    top3 = 0
    top5 = 0
    mrr = 0.0
    para = []
    record_correct: dict[str, list[bool]] = defaultdict(list)
    record_exec_probs: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        step = row["step"]
        pred, ranked = predict_weighted(row, weights)
        ranked_ids = [api.record_id for api in ranked]
        correct = pred.record_id == step.gold_api_id
        hits += int(correct)
        top3 += int(step.gold_api_id in ranked_ids[:3])
        top5 += int(step.gold_api_id in ranked_ids[:5])
        mrr += 1.0 / (ranked_ids.index(step.gold_api_id) + 1)
        para.append(param_f1(step, pred))
        record_correct[step.record_id].append(correct)
        record_exec_probs[step.record_id].append(
            simulated_step_success(step, pred, dependency_ok=row["dependency_ok"], retry_enabled=True)
        )

    workflows = len(record_correct)
    workflow_exec_probs = [math.prod(values) if values else 0.0 for values in record_exec_probs.values()]
    return {
        "api_acc": hits / len(rows) if rows else 0.0,
        "api_top3": top3 / len(rows) if rows else 0.0,
        "api_top5": top5 / len(rows) if rows else 0.0,
        "api_mrr": mrr / len(rows) if rows else 0.0,
        "workflow_exact": sum(1 for values in record_correct.values() if values and all(values)) / workflows
        if workflows
        else 0.0,
        "sim_exec_sr": sum(workflow_exec_probs) / len(workflow_exec_probs) if workflow_exec_probs else 0.0,
        "sim_step_exec_sr": sum(p for values in record_exec_probs.values() for p in values) / len(rows) if rows else 0.0,
        "para_f1": sum(para) / len(para) if para else 0.0,
        "steps": len(rows),
        "workflows": workflows,
    }


def weight_grid() -> list[dict[str, float]]:
    rows = []
    for rank, trace, service, operation, param, quality in product(
        [0.0, 0.1, 0.2, 0.35, 0.5],
        [0.0, 0.1, 0.25, 0.4, 0.6],
        [0.0, 0.05, 0.1, 0.2],
        [0.0, 0.05, 0.1, 0.2],
        [0.0, 0.05, 0.1, 0.2],
        [-0.10, -0.05, 0.0, 0.03, 0.06],
    ):
        rows.append(
            {
                "rank": rank,
                "sim": 1.0,
                "trace": trace,
                "service": service,
                "operation": operation,
                "param": param,
                "quality": quality,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    quality_by_id = load_emp_quality(args.emp_xlsx)
    records, steps, api_by_id = load_records(args.composition_data, quality_by_id)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    train_steps = [step for step in steps if step.record_id in splits["train"]]
    val_steps = [step for step in steps if step.record_id in splits["val"]]
    test_steps = [step for step in steps if step.record_id in splits["test"]]
    scorer = EmpScorer(
        train_steps,
        sorted(api_by_id.values(), key=lambda api: api.record_id),
        trace_top_k=args.trace_top_k,
    )
    val_rows = build_rows(scorer, val_steps, args.candidate_top_k)
    test_rows = build_rows(scorer, test_steps, args.candidate_top_k)

    rows = []
    for weights in weight_grid():
        val = evaluate(val_rows, weights)
        # API correctness is primary; workflow and simulated execution break ties.
        objective = val["api_acc"] + 0.10 * val["workflow_exact"] + 0.05 * val["sim_exec_sr"]
        rows.append({"weights": weights, "val": val, "objective": objective})
    rows.sort(
        key=lambda row: (
            row["objective"],
            row["val"]["api_acc"],
            row["val"]["workflow_exact"],
            row["val"]["sim_exec_sr"],
        ),
        reverse=True,
    )
    best = rows[0]
    best["test"] = evaluate(test_rows, best["weights"])
    payload = {
        "config": vars(args),
        "best_validation_adaptive_gems": best,
        "top_20": rows[:20],
        "notes": [
            "Weights are selected only on the validation split.",
            "This experiment is valid as a deployable adaptive scorer if reported with the fixed validation-selected weights.",
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
