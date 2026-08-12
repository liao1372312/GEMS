#!/usr/bin/env python
"""Evaluate a Reflexion-style textual experience memory baseline.

The baseline stores per-API textual reflections extracted from training traces.
At test time it retrieves reflections by TF-IDF similarity to the current
subtask and reranks the provided candidate APIs. It is intentionally not graph
structured and does not use reliability propagation, so it serves as a generic
experience-memory comparator for the main static-memory experiment.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_public_composition_experiments import (
    StepExample,
    candidate_text,
    endpoint_key,
    f1_score,
    load_composition_examples,
    required_param_names,
    split_records,
    step_text,
)


@dataclass
class Reflection:
    api_name: str
    endpoint_key: str
    text: str
    support: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--output", default="outputs/reflexion_memory_baseline.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--top-k-reflections", type=int, default=8)
    parser.add_argument("--sim-weight", type=float, default=0.70)
    parser.add_argument("--reflection-weight", type=float, default=0.25)
    parser.add_argument("--support-weight", type=float, default=0.05)
    parser.add_argument(
        "--preset",
        choices=["default", "semantic_safe"],
        default="default",
        help="Optional baseline preset. semantic_safe keeps AgentKB close to semantic retrieval while retaining reflection evidence.",
    )
    parser.add_argument(
        "--max-val-workflows",
        type=int,
        default=0,
        help="Limit validation workflows after the fixed split; 0 means all.",
    )
    parser.add_argument(
        "--max-test-workflows",
        type=int,
        default=0,
        help="Limit test workflows after the fixed split; 0 means all.",
    )
    return parser.parse_args()


def build_reflections(train_examples: list[StepExample]) -> list[Reflection]:
    grouped: dict[str, dict[str, Any]] = {}
    for example in train_examples:
        candidate = example.gold_candidate
        key = endpoint_key(candidate) or str(candidate.get("api_name") or "")
        api_name = str(candidate.get("api_name") or "")
        row = grouped.setdefault(
            key,
            {
                "api_name": api_name,
                "endpoint_key": key,
                "support": 0,
                "texts": [],
            },
        )
        row["support"] += 1
        row["texts"].append(
            " ".join(
                [
                    "Successful API experience.",
                    f"Request: {example.user_query}",
                    f"Subtask: {example.step_description}",
                    f"Domain: {example.record_domain}",
                    f"API: {api_name}",
                    f"Candidate: {candidate_text(candidate)}",
                ]
            )
        )
    reflections: list[Reflection] = []
    for row in grouped.values():
        reflections.append(
            Reflection(
                api_name=row["api_name"],
                endpoint_key=row["endpoint_key"],
                text=" ".join(row["texts"][:8]),
                support=int(row["support"]),
            )
        )
    return reflections


def evaluate(
    examples: list[StepExample],
    reflections: list[Reflection],
    *,
    top_k_reflections: int,
    sim_weight: float,
    reflection_weight: float,
    support_weight: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    docs = [reflection.text or "empty" for reflection in reflections] or ["empty"]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=50000,
    )
    matrix = vectorizer.fit_transform(docs)

    step_hits = 0
    step_top3 = 0
    reciprocal_sum = 0.0
    param_f1_values: list[float] = []
    workflows: dict[str, list[bool]] = defaultdict(list)

    for example in examples:
        query = vectorizer.transform([step_text(example) or "empty"])
        sims = cosine_similarity(query, matrix).ravel()
        top_indices = sorted(range(len(reflections)), key=lambda idx: sims[idx], reverse=True)[:top_k_reflections]
        reflection_scores: dict[str, float] = defaultdict(float)
        max_support = max((reflections[idx].support for idx in top_indices), default=1)
        for idx in top_indices:
            reflection = reflections[idx]
            score = float(sims[idx])
            reflection_scores[reflection.endpoint_key] = max(reflection_scores[reflection.endpoint_key], score)
            if reflection.api_name:
                reflection_scores[reflection.api_name] = max(reflection_scores[reflection.api_name], 0.95 * score)

        scores: list[float] = []
        for candidate in example.candidates:
            semantic = float(candidate.get("similarity_score") or 0.0)
            key = endpoint_key(candidate)
            api_name = str(candidate.get("api_name") or "")
            memory_score = max(reflection_scores.get(key, 0.0), reflection_scores.get(api_name, 0.0))
            support = 0.0
            for idx in top_indices:
                reflection = reflections[idx]
                if reflection.endpoint_key == key or reflection.api_name == api_name:
                    support = max(support, reflection.support / max_support if max_support else 0.0)
            scores.append(sim_weight * semantic + reflection_weight * memory_score + support_weight * support)

        ranked = sorted(range(len(scores)), key=lambda idx: (scores[idx], -idx), reverse=True)
        pred_index = ranked[0] if ranked else 0
        correct = pred_index == example.gold_index
        step_hits += int(correct)
        step_top3 += int(example.gold_index in ranked[:3])
        reciprocal_sum += 1.0 / (ranked.index(example.gold_index) + 1)
        workflows[example.record_id].append(correct)
        param_f1_values.append(
            f1_score(
                required_param_names(example.candidates[pred_index]),
                required_param_names(example.gold_candidate),
            )
        )

    steps = len(examples)
    return {
        "api_acc": step_hits / steps if steps else 0.0,
        "api_top3": step_top3 / steps if steps else 0.0,
        "api_mrr": reciprocal_sum / steps if steps else 0.0,
        "workflow_exact": sum(1 for values in workflows.values() if values and all(values)) / len(workflows) if workflows else 0.0,
        "para_f1": statistics.mean(param_f1_values) if param_f1_values else 0.0,
        "hallu_rate": 0.0,
        "steps": steps,
        "workflows": len(workflows),
        "avg_latency_ms_per_step": 1000.0 * (time.perf_counter() - start) / steps if steps else 0.0,
    }


def limit_workflows(examples: list[StepExample], max_workflows: int) -> list[StepExample]:
    if max_workflows <= 0:
        return examples
    selected: list[StepExample] = []
    seen: set[str] = set()
    for example in examples:
        if example.record_id not in seen:
            if len(seen) >= max_workflows:
                break
            seen.add(example.record_id)
        selected.append(example)
    return selected


def main() -> None:
    args = parse_args()
    if args.preset == "semantic_safe":
        args.top_k_reflections = 8
        args.sim_weight = 1.0
        args.reflection_weight = 0.05
        args.support_weight = 0.0
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    train_examples = [example for example in examples if example.record_id in splits["train"]]
    val_examples = [example for example in examples if example.record_id in splits["val"]]
    test_examples = [example for example in examples if example.record_id in splits["test"]]
    eval_val_examples = limit_workflows(val_examples, args.max_val_workflows)
    eval_test_examples = limit_workflows(test_examples, args.max_test_workflows)

    reflections = build_reflections(train_examples)
    payload = {
        "config": vars(args),
        "paper_ready": True,
        "selection_split": "fixed",
        "note": "Reflexion-Memory uses training-trace textual reflections only; no graph reliability or test-time update.",
        "memory": {
            "reflections": len(reflections),
            "train_steps": len(train_examples),
        },
        "results": {
            "val": evaluate(
                eval_val_examples,
                reflections,
                top_k_reflections=args.top_k_reflections,
                sim_weight=args.sim_weight,
                reflection_weight=args.reflection_weight,
                support_weight=args.support_weight,
            ),
            "test": evaluate(
                eval_test_examples,
                reflections,
                top_k_reflections=args.top_k_reflections,
                sim_weight=args.sim_weight,
                reflection_weight=args.reflection_weight,
                support_weight=args.support_weight,
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["results"], ensure_ascii=False, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
