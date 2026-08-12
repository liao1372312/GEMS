#!/usr/bin/env python
"""Workflow-level semantic-risk router for plan accuracy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_semantic_risk_llm_router import features as step_features
from evaluate_semantic_risk_llm_router import load_llm_predictions, key
from llm_public_composition_rerank import selected_index_from_result
from run_public_composition_experiments import (
    f1_score,
    load_composition_examples,
    required_param_names,
    split_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--llm-files", nargs="*", default=[])
    parser.add_argument("--output", default="outputs/workflow_risk_llm_router_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def by_record(examples: list[Any]) -> dict[str, list[Any]]:
    rows: dict[str, list[Any]] = {}
    for example in examples:
        rows.setdefault(example.record_id, []).append(example)
    return {key_: sorted(value, key=lambda item: item.step_id) for key_, value in rows.items()}


def workflow_features(plan: list[Any]) -> list[float]:
    matrix = np.asarray([step_features(example) for example in plan], dtype=float)
    margins = matrix[:, 3]
    top1 = matrix[:, 0]
    param_overlap = matrix[:, 10]
    return [
        float(len(plan)),
        float(np.mean(top1)),
        float(np.min(top1)),
        float(np.mean(margins)),
        float(np.min(margins)),
        float(np.std(margins)),
        float(np.mean(param_overlap)),
        float(np.min(param_overlap)),
        float(np.mean([example.gold_index != 0 for example in plan])),  # train/val only leakage; removed below
    ]


def safe_workflow_features(plan: list[Any]) -> list[float]:
    values = workflow_features(plan)
    return values[:-1]


def build_matrix(records: dict[str, list[Any]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ids = []
    x = []
    y = []
    for record_id, plan in records.items():
        ids.append(record_id)
        x.append(safe_workflow_features(plan))
        y.append(int(not all(example.gold_index == 0 for example in plan)))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=int), ids


def confidence(pred: dict[str, Any] | None) -> float:
    if not pred:
        return 0.0
    try:
        return float((pred.get("parsed") or {}).get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def evaluate(
    plans: dict[str, list[Any]],
    predictions: dict[str, dict[str, Any]],
    model: Any,
    threshold: float,
    accept_mode: str,
) -> dict[str, Any]:
    x = np.asarray([safe_workflow_features(plan) for plan in plans.values()], dtype=float)
    risk = model.predict_proba(x)[:, 1]
    hits = 0
    workflow_hits = 0
    para = []
    llm_plans = 0
    llm_changed = 0
    steps = 0
    for (record_id, plan), plan_risk in zip(plans.items(), risk):
        use_llm_plan = plan_risk >= threshold
        if use_llm_plan and accept_mode == "high_conf":
            confs = [confidence(predictions.get(key(example.record_id, example.step_id))) for example in plan]
            use_llm_plan = (sum(confs) / len(confs)) >= 0.85
        llm_plans += int(use_llm_plan)
        plan_ok = True
        for example in plan:
            pred_index = 0
            if use_llm_plan:
                pred = predictions.get(key(example.record_id, example.step_id))
                if pred:
                    pred_index = selected_index_from_result(pred, len(example.candidates))
                    llm_changed += int(pred_index != 0)
            correct = pred_index == example.gold_index
            hits += int(correct)
            plan_ok = plan_ok and correct
            para.append(
                f1_score(
                    required_param_names(example.candidates[pred_index]),
                    required_param_names(example.gold_candidate),
                )
            )
            steps += 1
        workflow_hits += int(plan_ok)
    return {
        "api_acc": hits / steps if steps else 0.0,
        "workflow_exact": workflow_hits / len(plans) if plans else 0.0,
        "para_f1": sum(para) / len(para) if para else 0.0,
        "llm_plan_rate": llm_plans / len(plans) if plans else 0.0,
        "llm_changed_rate": llm_changed / steps if steps else 0.0,
        "steps": steps,
        "workflows": len(plans),
    }


class FixedModel:
    def __init__(self, value: float, feature_count: int) -> None:
        self.value = value
        self.feature_count = feature_count

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.full(len(x), self.value)
        return np.column_stack([1.0 - p, p])


def main() -> None:
    args = parse_args()
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    split_examples = {
        split: [example for example in examples if example.record_id in ids]
        for split, ids in splits.items()
    }
    train_plans = by_record(split_examples["train"])
    val_plans = by_record(split_examples["val"])
    test_plans = by_record(split_examples["test"])
    x_train, y_train, _ = build_matrix(train_plans)
    x_val, y_val, _ = build_matrix(val_plans)
    predictions = load_llm_predictions(args.llm_files)

    model_candidates = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)),
        "rf": RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=6, class_weight="balanced", random_state=args.seed, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(max_iter=120, learning_rate=0.04, max_leaf_nodes=7, l2_regularization=0.05, random_state=args.seed),
    }
    rows = []
    thresholds = np.linspace(0.0, 1.0, 41).tolist()
    for name, model in model_candidates.items():
        model.fit(x_train, y_train)
        val_scores = model.predict_proba(x_val)[:, 1]
        for threshold in sorted(set(thresholds + np.quantile(val_scores, np.linspace(0, 1, 21)).tolist())):
            pred = val_scores >= threshold
            recall = float(((pred == 1) & (y_val == 1)).sum() / (y_val == 1).sum())
            route_rate = float(pred.mean())
            # Validation objective estimates plan preservation: catch risky plans
            # but avoid routing too many complete workflows to LLM.
            objective = recall - 0.55 * route_rate
            rows.append({"name": name, "model": model, "threshold": float(threshold), "objective": objective, "val_recall": recall, "val_route_rate": route_rate})
    rows.sort(key=lambda row: (row["objective"], row["val_recall"], -row["val_route_rate"]), reverse=True)
    best = rows[0]
    test = evaluate(test_plans, predictions, best["model"], best["threshold"], "none")
    test_conf = evaluate(test_plans, predictions, best["model"], best["threshold"], "high_conf")
    feature_count = x_train.shape[1]
    payload = {
        "best": {k: v for k, v in best.items() if k != "model"},
        "workflow_risk_router_test": test,
        "workflow_risk_high_conf_router_test": test_conf,
        "semantic_top1_test": evaluate(test_plans, predictions, FixedModel(0.0, feature_count), 1.0, "none"),
        "all_llm_test": evaluate(test_plans, predictions, FixedModel(1.0, feature_count), 0.0, "none"),
        "top_10": [{k: v for k, v in row.items() if k != "model"} for row in rows[:10]],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
