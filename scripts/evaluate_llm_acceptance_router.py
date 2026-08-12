#!/usr/bin/env python
"""Evaluate a validation-trained LLM acceptance router.

The router first obtains an LLM reranking result, then accepts it only when a
validation-trained acceptance model predicts that the LLM will improve over the
semantic top-1 candidate. If full validation LLM predictions are unavailable,
the script explicitly marks the result as not paper-ready.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_public_composition_rerank import selected_index_from_result
from run_public_composition_experiments import (
    candidate_text,
    f1_score,
    load_composition_examples,
    required_param_names,
    split_records,
    step_text,
)
from gems.text import normalize_text


FEATURE_NAMES = [
    "llm_rank",
    "llm_rank_recip",
    "llm_confidence",
    "top1_similarity",
    "llm_similarity",
    "similarity_drop",
    "margin_1_2",
    "margin_1_3",
    "llm_param_overlap",
    "top1_param_overlap",
    "llm_api_query_overlap",
    "top1_api_query_overlap",
    "llm_text_query_overlap",
    "top1_text_query_overlap",
    "same_domain_llm",
    "same_domain_top1",
    "plan_steps",
    "step_id_norm",
]


@dataclass
class StepRow:
    example: Any
    features: list[float]
    semantic_index: int
    llm_index: int
    gold_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--llm-files", nargs="*", default=[])
    parser.add_argument("--output", default="outputs/llm_acceptance_router_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def key(record_id: str, step_id: int) -> str:
    return f"{record_id}::{step_id}"


def load_llm_predictions(files: list[str]) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    paths = [Path(name) for name in files] if files else sorted(Path("outputs").glob("llm_public_composition_rerank*.json"))
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
    parsed = prediction.get("parsed") or {}
    try:
        return float(parsed.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def token_set(text: object) -> set[str]:
    return {token for token in normalize_text(text).replace("/", " ").replace("-", " ").replace("_", " ").split() if token}


def overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def param_overlap(example: Any, candidate_index: int) -> float:
    required = token_set(" ".join(example.required_inputs))
    if not required:
        return 1.0
    params = token_set(" ".join(required_param_names(example.candidates[candidate_index])))
    if not params:
        return 0.0
    return len(required & params) / len(required)


def build_rows(examples: list[Any], predictions: dict[str, dict[str, Any]]) -> list[StepRow]:
    plan_lengths: dict[str, int] = {}
    for example in examples:
        plan_lengths[example.record_id] = plan_lengths.get(example.record_id, 0) + 1

    rows: list[StepRow] = []
    for example in examples:
        pred = predictions.get(key(example.record_id, example.step_id))
        if not pred:
            continue
        llm_index = selected_index_from_result(pred, len(example.candidates))
        scores = [float(candidate.get("similarity_score") or 0.0) for candidate in example.candidates]
        padded = scores + [0.0, 0.0, 0.0]
        top1 = example.candidates[0]
        llm_candidate = example.candidates[llm_index]
        query_tokens = token_set(step_text(example))
        top1_api_tokens = token_set(top1.get("api_name"))
        llm_api_tokens = token_set(llm_candidate.get("api_name"))
        top1_text_tokens = token_set(candidate_text(top1))
        llm_text_tokens = token_set(candidate_text(llm_candidate))
        record_domain = str(example.record_domain or "").lower()
        top1_domain = str(top1.get("domain") or "").lower()
        llm_domain = str(llm_candidate.get("domain") or "").lower()
        plan_steps = plan_lengths[example.record_id]
        features = [
            float(llm_index + 1),
            1.0 / float(llm_index + 1),
            confidence(pred),
            padded[0],
            scores[llm_index] if llm_index < len(scores) else 0.0,
            padded[0] - (scores[llm_index] if llm_index < len(scores) else 0.0),
            padded[0] - padded[1],
            padded[0] - padded[2],
            param_overlap(example, llm_index),
            param_overlap(example, 0),
            overlap(query_tokens, llm_api_tokens),
            overlap(query_tokens, top1_api_tokens),
            overlap(query_tokens, llm_text_tokens),
            overlap(query_tokens, top1_text_tokens),
            float(bool(record_domain and record_domain in llm_domain)),
            float(bool(record_domain and record_domain in top1_domain)),
            float(plan_steps),
            float(example.step_id) / float(max(plan_steps, 1)),
        ]
        rows.append(StepRow(example, features, 0, llm_index, example.gold_index))
    return rows


def evaluate_indices(rows: list[StepRow], chosen: dict[tuple[str, int], int]) -> dict[str, Any]:
    by_record: dict[str, list[StepRow]] = {}
    for row in rows:
        by_record.setdefault(row.example.record_id, []).append(row)
    hits = 0
    workflow_hits = 0
    para_f1_values: list[float] = []
    accepted = 0
    changed = 0
    for plan in by_record.values():
        ok = True
        for row in sorted(plan, key=lambda item: item.example.step_id):
            idx = chosen.get((row.example.record_id, row.example.step_id), 0)
            accepted += int(idx == row.llm_index)
            changed += int(idx != 0)
            correct = idx == row.gold_index
            hits += int(correct)
            ok = ok and correct
            para_f1_values.append(
                f1_score(
                    required_param_names(row.example.candidates[idx]),
                    required_param_names(row.example.gold_candidate),
                )
            )
        workflow_hits += int(ok)
    steps = len(rows)
    workflows = len(by_record)
    return {
        "api_acc": hits / steps if steps else 0.0,
        "workflow_exact": workflow_hits / workflows if workflows else 0.0,
        "para_f1": sum(para_f1_values) / len(para_f1_values) if para_f1_values else 0.0,
        "llm_accept_rate": accepted / steps if steps else 0.0,
        "llm_change_rate": changed / steps if steps else 0.0,
        "steps": steps,
        "workflows": workflows,
    }


def evaluate_confidence_threshold(
    examples: list[Any],
    predictions: dict[str, dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    hits = 0
    workflows: dict[str, list[bool]] = {}
    para_f1: list[float] = []
    llm_available = 0
    llm_accepted = 0
    for example in examples:
        pred = predictions.get(key(example.record_id, example.step_id))
        pred_index = 0
        if pred:
            llm_available += 1
            if confidence(pred) >= threshold:
                llm_accepted += 1
                pred_index = selected_index_from_result(pred, len(example.candidates))
        correct = pred_index == example.gold_index
        hits += int(correct)
        workflows.setdefault(example.record_id, []).append(correct)
        para_f1.append(
            f1_score(
                required_param_names(example.candidates[pred_index]),
                required_param_names(example.gold_candidate),
            )
        )
    n = len(examples)
    return {
        "api_acc": hits / n if n else 0.0,
        "workflow_exact": sum(1 for values in workflows.values() if all(values)) / len(workflows) if workflows else 0.0,
        "para_f1": sum(para_f1) / len(para_f1) if para_f1 else 0.0,
        "llm_available": llm_available,
        "llm_accepted": llm_accepted,
        "llm_accept_rate": llm_accepted / n if n else 0.0,
        "coverage": llm_available / n if n else 0.0,
        "steps": n,
    }


def train_acceptance_model(rows: list[StepRow], seed: int) -> Any:
    x_train = np.asarray([row.features for row in rows], dtype=float)
    y_train = np.asarray(
        [int(row.llm_index == row.gold_index and row.semantic_index != row.gold_index) for row in rows],
        dtype=int,
    )
    if len(set(y_train.tolist())) < 2:
        return None
    models = [
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        HistGradientBoostingClassifier(
            max_iter=140,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.04,
            random_state=seed,
        ),
    ]
    fitted = []
    for model in models:
        model.fit(x_train, y_train)
        fitted.append(model)
    return fitted


def predict_acceptance_probs(models: list[Any] | None, rows: list[StepRow]) -> dict[tuple[str, int], float]:
    if not models or not rows:
        return {(row.example.record_id, row.example.step_id): 0.0 for row in rows}
    x = np.asarray([row.features for row in rows], dtype=float)
    probs = np.zeros(len(rows), dtype=float)
    for model in models:
        if hasattr(model, "predict_proba"):
            probs += model.predict_proba(x)[:, 1]
        else:
            probs += model.decision_function(x)
    probs /= len(models)
    return {(row.example.record_id, row.example.step_id): float(prob) for row, prob in zip(rows, probs)}


def choose_with_threshold(
    rows: list[StepRow],
    probs: dict[tuple[str, int], float],
    threshold: float,
) -> dict[tuple[str, int], int]:
    chosen: dict[tuple[str, int], int] = {}
    for row in rows:
        step_key = (row.example.record_id, row.example.step_id)
        chosen[step_key] = row.llm_index if probs.get(step_key, 0.0) >= threshold else 0
    return chosen


def train_val_acceptance(
    val_rows: list[StepRow],
    test_rows: list[StepRow],
    seed: int,
) -> dict[str, Any]:
    models = train_acceptance_model(val_rows, seed)
    val_probs = predict_acceptance_probs(models, val_rows)
    test_probs = predict_acceptance_probs(models, test_rows)
    thresholds = [round(value, 3) for value in np.linspace(0.05, 0.95, 37)]
    sweep = []
    for threshold in thresholds:
        metrics = evaluate_indices(val_rows, choose_with_threshold(val_rows, val_probs, threshold))
        balanced = 0.35 * metrics["api_acc"] + 0.40 * metrics["workflow_exact"] + 0.25 * metrics["para_f1"]
        sweep.append({"threshold": threshold, "metrics": metrics, "balanced": balanced})
    sweep.sort(key=lambda item: (item["balanced"], item["metrics"]["workflow_exact"], item["metrics"]["api_acc"]), reverse=True)
    best = sweep[0]
    test = evaluate_indices(test_rows, choose_with_threshold(test_rows, test_probs, best["threshold"]))
    return {
        "best_threshold": best["threshold"],
        "val": best["metrics"],
        "test": test,
        "val_sweep_top10": sweep[:10],
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
    val_rows_model = build_rows(split_examples["val"], predictions)
    test_rows_model = build_rows(split_examples["test"], predictions)
    thresholds = [0.0, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99, 1.01]
    val_rows = [{"threshold": t, "metrics": evaluate_confidence_threshold(split_examples["val"], predictions, t)} for t in thresholds]
    val_has_full_coverage = any(row["metrics"]["coverage"] >= 0.99 for row in val_rows)
    if val_has_full_coverage:
        best = max(val_rows, key=lambda row: (row["metrics"]["api_acc"], row["metrics"]["workflow_exact"], row["metrics"]["para_f1"]))
        selection_note = "threshold selected on validation split"
    else:
        # Diagnostic only: useful before the user runs validation LLM.
        test_rows = [{"threshold": t, "metrics": evaluate_confidence_threshold(split_examples["test"], predictions, t)} for t in thresholds]
        best = max(test_rows, key=lambda row: (row["metrics"]["api_acc"], row["metrics"]["workflow_exact"], row["metrics"]["para_f1"]))
        selection_note = "diagnostic only: validation LLM coverage is incomplete, so threshold was selected on test"

    test = evaluate_confidence_threshold(split_examples["test"], predictions, best["threshold"])
    semantic = evaluate_confidence_threshold(split_examples["test"], predictions, 1.01)
    all_llm = evaluate_confidence_threshold(split_examples["test"], predictions, 0.0)
    formal_model = train_val_acceptance(val_rows_model, test_rows_model, args.seed) if val_has_full_coverage else None
    payload = {
        "paper_ready": bool(val_has_full_coverage),
        "best_threshold": best["threshold"],
        "selection_note": selection_note,
        "semantic_top1_test": semantic,
        "all_llm_test": all_llm,
        "router_test": test,
        "validation_trained_acceptance": formal_model,
        "val_sweep": val_rows,
        "available_llm_predictions": len(predictions),
        "coverage": {
            "val_available": len(val_rows_model),
            "val_steps": len(split_examples["val"]),
            "val_coverage": len(val_rows_model) / len(split_examples["val"]) if split_examples["val"] else 0.0,
            "test_available": len(test_rows_model),
            "test_steps": len(split_examples["test"]),
            "test_coverage": len(test_rows_model) / len(split_examples["test"]) if split_examples["test"] else 0.0,
        },
        "feature_names": FEATURE_NAMES,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
