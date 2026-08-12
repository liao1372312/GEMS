#!/usr/bin/env python
"""Cross-validated LLM acceptance model on the 211-workflow test diagnostic set."""

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
    parser.add_argument("--llm-result", default="outputs/llm_public_composition_rerank_test_all.json")
    parser.add_argument("--output", default="outputs/test211_cv_acceptance.json")
    parser.add_argument("--markdown-output", default="outputs/test211_cv_acceptance.md")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def key(record_id: str, step_id: int) -> str:
    return f"{record_id}::{step_id}"


def token_set(text: object) -> set[str]:
    return {token for token in normalize_text(text).replace("/", " ").replace("-", " ").replace("_", " ").split() if token}


def overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def confidence(prediction: dict[str, Any] | None) -> float:
    if not prediction:
        return 0.0
    try:
        return float((prediction.get("parsed") or {}).get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


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


def group_ids(rows: list[StepRow]) -> list[str]:
    return sorted({row.example.record_id for row in rows})


def split_folds(ids: list[str], folds: int, seed: int) -> list[set[str]]:
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    buckets = [set() for _ in range(folds)]
    for idx, record_id in enumerate(ids):
        buckets[idx % folds].add(record_id)
    return buckets


def evaluate_indices(rows: list[StepRow], chosen: dict[tuple[str, int], int]) -> dict[str, Any]:
    by_record: dict[str, list[StepRow]] = {}
    for row in rows:
        by_record.setdefault(row.example.record_id, []).append(row)
    hits = 0
    workflow_hits = 0
    para_f1_values: list[float] = []
    changed = 0
    accepted = 0
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


def fit_predict_acceptance(train_rows: list[StepRow], test_rows: list[StepRow], seed: int) -> dict[tuple[str, int], float]:
    x_train = np.asarray([row.features for row in train_rows], dtype=float)
    # Label accepts only LLM-help cases. LLM-hurt and both-wrong cases are reject.
    y_train = np.asarray([int(row.llm_index == row.gold_index and row.semantic_index != row.gold_index) for row in train_rows], dtype=int)
    x_test = np.asarray([row.features for row in test_rows], dtype=float)
    models = [
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=8, class_weight="balanced", random_state=seed, n_jobs=-1),
        HistGradientBoostingClassifier(max_iter=140, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.04, random_state=seed),
    ]
    probs = np.zeros(len(test_rows), dtype=float)
    for model in models:
        model.fit(x_train, y_train)
        if hasattr(model, "predict_proba"):
            probs += model.predict_proba(x_test)[:, 1]
        else:
            probs += model.decision_function(x_test)
    probs /= len(models)
    return {(row.example.record_id, row.example.step_id): float(prob) for row, prob in zip(test_rows, probs)}


def choose_with_threshold(rows: list[StepRow], probs: dict[tuple[str, int], float], threshold: float) -> dict[tuple[str, int], int]:
    chosen: dict[tuple[str, int], int] = {}
    for row in rows:
        step_key = (row.example.record_id, row.example.step_id)
        chosen[step_key] = row.llm_index if probs.get(step_key, 0.0) >= threshold else 0
    return chosen


def main() -> None:
    args = parse_args()
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    test_examples = [example for example in examples if example.record_id in splits["test"]]
    predictions = load_predictions(args.llm_result)
    rows = build_rows(test_examples, predictions)

    ids = group_ids(rows)
    folds = split_folds(ids, args.folds, args.seed)
    all_probs: dict[tuple[str, int], float] = {}
    fold_reports = []
    for fold_idx, heldout_ids in enumerate(folds):
        train_rows = [row for row in rows if row.example.record_id not in heldout_ids]
        heldout_rows = [row for row in rows if row.example.record_id in heldout_ids]
        probs = fit_predict_acceptance(train_rows, heldout_rows, args.seed + fold_idx)
        all_probs.update(probs)
        fold_reports.append({"fold": fold_idx, "workflows": len(heldout_ids), "steps": len(heldout_rows)})

    thresholds = [round(value, 3) for value in np.linspace(0.05, 0.95, 37)]
    sweep = []
    for threshold in thresholds:
        metrics = evaluate_indices(rows, choose_with_threshold(rows, all_probs, threshold))
        balanced = 0.35 * metrics["api_acc"] + 0.40 * metrics["workflow_exact"] + 0.25 * metrics["para_f1"]
        sweep.append({"threshold": threshold, "metrics": metrics, "balanced": balanced})
    sweep.sort(key=lambda item: (item["balanced"], item["metrics"]["workflow_exact"], item["metrics"]["api_acc"]), reverse=True)
    best = sweep[0]

    semantic = evaluate_indices(rows, {(row.example.record_id, row.example.step_id): 0 for row in rows})
    all_llm = evaluate_indices(rows, {(row.example.record_id, row.example.step_id): row.llm_index for row in rows})
    oracle = evaluate_indices(
        rows,
        {
            (row.example.record_id, row.example.step_id): (
                row.llm_index if row.llm_index == row.gold_index and row.semantic_index != row.gold_index else 0
            )
            for row in rows
        },
    )
    payload = {
        "diagnostic_only": True,
        "reason": "Cross-validation is performed inside the 211-workflow test split because full validation LLM predictions are unavailable.",
        "feature_names": FEATURE_NAMES,
        "folds": fold_reports,
        "semantic_top1": semantic,
        "all_llm": all_llm,
        "oracle_semantic_or_llm": oracle,
        "best_cv_acceptance": best,
        "top_10": sweep[:10],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(Path(args.markdown_output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved {output}")
    print(f"saved {args.markdown_output}")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = [
        ("Semantic top-1", payload["semantic_top1"]),
        ("Full LLM", payload["all_llm"]),
        ("CV-Accept", payload["best_cv_acceptance"]["metrics"]),
        ("Oracle Sem/LLM", payload["oracle_semantic_or_llm"]),
    ]
    lines = [
        "# Test-211 Cross-Validated Acceptance",
        "",
        "Diagnostic only: folds are created inside the 211-workflow test split.",
        "",
        "| Method | API.Acc | Plan.Acc | Para.F1 | LLM Change |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in rows:
        lines.append(
            f"| {name} | {metrics['api_acc']:.4f} | {metrics['workflow_exact']:.4f} | "
            f"{metrics['para_f1']:.4f} | {metrics['llm_change_rate']:.4f} |"
        )
    lines.append("")
    lines.append(f"Best CV threshold: `{payload['best_cv_acceptance']['threshold']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
