#!/usr/bin/env python
"""Route high semantic-risk steps to the LLM reranker.

The risk model is trained only from train/validation labels that indicate
whether semantic top-1 is correct. At test time, high-risk steps use cached LLM
reranking predictions; low-risk steps keep semantic top-1.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score as binary_f1
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
    "top1_similarity",
    "top2_similarity",
    "top3_similarity",
    "margin_1_2",
    "margin_1_3",
    "score_mean",
    "score_std",
    "score_range",
    "top1_param_count",
    "top1_has_params",
    "top1_required_input_overlap",
    "top1_api_name_overlap",
    "top1_text_overlap",
    "top1_domain_overlap",
    "candidate_count",
]


@dataclass
class RiskModelResult:
    name: str
    model: Any
    threshold: float
    val_f1: float
    val_wrong_recall: float
    val_route_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--llm-files", nargs="*", default=[])
    parser.add_argument("--output", default="outputs/semantic_risk_llm_router_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
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


def token_set(text: object) -> set[str]:
    return {tok for tok in normalize_text(text).replace("/", " ").replace("-", " ").split() if tok}


def overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def features(example: Any) -> list[float]:
    scores = [float(candidate.get("similarity_score") or 0.0) for candidate in example.candidates]
    padded = scores + [0.0, 0.0, 0.0]
    top1 = example.candidates[0]
    query_tokens = token_set(step_text(example))
    required_tokens = token_set(" ".join(example.required_inputs))
    top_params = required_param_names(top1)
    param_tokens = token_set(" ".join(top_params))
    api_tokens = token_set(top1.get("api_name"))
    text_tokens = token_set(candidate_text(top1))
    domain_tokens = token_set(top1.get("domain"))
    return [
        padded[0],
        padded[1],
        padded[2],
        padded[0] - padded[1],
        padded[0] - padded[2],
        float(np.mean(scores)) if scores else 0.0,
        float(np.std(scores)) if scores else 0.0,
        (max(scores) - min(scores)) if scores else 0.0,
        float(len(top_params)),
        float(bool(top_params)),
        overlap(required_tokens, param_tokens),
        overlap(query_tokens, api_tokens),
        overlap(query_tokens, text_tokens),
        overlap(query_tokens, domain_tokens),
        float(len(example.candidates)),
    ]


def build_matrix(examples: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([features(example) for example in examples], dtype=float)
    # Label 1 means semantic top-1 is wrong and should be routed to LLM.
    y = np.asarray([int(example.gold_index != 0) for example in examples], dtype=int)
    return x, y


def risk_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return np.asarray(model.decision_function(x), dtype=float)


def tune_threshold(model: Any, x_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float, float, float]:
    scores = risk_scores(model, x_val)
    thresholds = sorted(set(np.quantile(scores, np.linspace(0.0, 1.0, 51)).tolist() + [0.0, 0.25, 0.5, 0.75, 1.0]))
    best = (0.0, -1.0, 0.0, 0.0)
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        f1 = binary_f1(y_val, pred, zero_division=0)
        wrong = y_val == 1
        recall = float(((pred == 1) & wrong).sum() / wrong.sum()) if wrong.sum() else 0.0
        route_rate = float(pred.mean()) if len(pred) else 0.0
        # Prefer useful wrong-step recall, but heavily penalize broad routing
        # because plan accuracy is damaged when correct semantic steps are
        # unnecessarily rewritten by the LLM.
        objective = 0.50 * recall + 0.50 * f1 - 0.35 * route_rate
        if objective > best[1]:
            best = (float(threshold), float(objective), float(recall), float(route_rate))
    threshold, _, recall, route_rate = best
    pred = (scores >= threshold).astype(int)
    return threshold, binary_f1(y_val, pred, zero_division=0), recall, route_rate


def train_models(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, seed: int) -> list[RiskModelResult]:
    candidates = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.03,
            random_state=seed,
        ),
    }
    results = []
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        threshold, f1, recall, route_rate = tune_threshold(model, x_val, y_val)
        results.append(RiskModelResult(name, model, threshold, f1, recall, route_rate))
    results.sort(key=lambda row: (row.val_wrong_recall, row.val_f1, -row.val_route_rate), reverse=True)
    return results


def evaluate_router(examples: list[Any], predictions: dict[str, dict[str, Any]], model: Any, threshold: float) -> dict[str, Any]:
    x = np.asarray([features(example) for example in examples], dtype=float)
    risks = risk_scores(model, x)
    by_record: dict[str, list[tuple[Any, float]]] = {}
    for example, risk in zip(examples, risks):
        by_record.setdefault(example.record_id, []).append((example, float(risk)))

    hits = 0
    para = []
    workflow_hits = 0
    llm_calls = 0
    llm_changed = 0
    missing_llm = 0
    for rows in by_record.values():
        plan_ok = True
        for example, risk in sorted(rows, key=lambda item: item[0].step_id):
            use_llm = risk >= threshold
            pred_index = 0
            if use_llm:
                llm_calls += 1
                pred = predictions.get(key(example.record_id, example.step_id))
                if pred:
                    pred_index = selected_index_from_result(pred, len(example.candidates))
                    llm_changed += int(pred_index != 0)
                else:
                    missing_llm += 1
            correct = pred_index == example.gold_index
            hits += int(correct)
            plan_ok = plan_ok and correct
            para.append(
                f1_score(
                    required_param_names(example.candidates[pred_index]),
                    required_param_names(example.gold_candidate),
                )
            )
        workflow_hits += int(plan_ok)

    n = len(examples)
    return {
        "api_acc": hits / n if n else 0.0,
        "workflow_exact": workflow_hits / len(by_record) if by_record else 0.0,
        "para_f1": sum(para) / len(para) if para else 0.0,
        "llm_calls": llm_calls,
        "llm_call_rate": llm_calls / n if n else 0.0,
        "llm_changed": llm_changed,
        "llm_change_rate": llm_changed / n if n else 0.0,
        "missing_llm": missing_llm,
        "steps": n,
        "workflows": len(by_record),
    }


def evaluate_fixed(examples: list[Any], predictions: dict[str, dict[str, Any]], mode: str) -> dict[str, Any]:
    class FixedModel:
        def __init__(self, value: float) -> None:
            self.value = value

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            p = np.full(len(x), self.value)
            return np.column_stack([1 - p, p])

    if mode == "semantic":
        return evaluate_router(examples, predictions, FixedModel(0.0), 1.0)
    return evaluate_router(examples, predictions, FixedModel(1.0), 0.0)


def main() -> None:
    args = parse_args()
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    split_examples = {
        split: [example for example in examples if example.record_id in ids]
        for split, ids in splits.items()
    }
    x_train, y_train = build_matrix(split_examples["train"])
    x_val, y_val = build_matrix(split_examples["val"])
    models = train_models(x_train, y_train, x_val, y_val, args.seed)
    predictions = load_llm_predictions(args.llm_files)

    rows = []
    for result in models:
        test = evaluate_router(split_examples["test"], predictions, result.model, result.threshold)
        rows.append(
            {
                "name": result.name,
                "threshold": result.threshold,
                "val_risk_f1": result.val_f1,
                "val_wrong_recall": result.val_wrong_recall,
                "val_route_rate": result.val_route_rate,
                "test": test,
            }
        )

    # Select by validation risk quality only, not test metrics.
    best = rows[0]
    payload = {
        "config": vars(args),
        "feature_names": FEATURE_NAMES,
        "best_model": best,
        "all_models": rows,
        "semantic_top1_test": evaluate_fixed(split_examples["test"], predictions, "semantic"),
        "all_llm_test": evaluate_fixed(split_examples["test"], predictions, "llm"),
        "notes": [
            "Risk model is trained on train split and threshold-tuned on validation semantic-error labels.",
            "Test LLM predictions are used only for evaluating routed high-risk steps.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
