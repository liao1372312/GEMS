#!/usr/bin/env python3
"""External validation on the HSC AI service composition dataset.

The HSC dataset provides natural-language requirements, gold functional
workflows, QoS-normalized Hugging Face services, and optimizer-generated best
solutions. This script evaluates service selection under the gold functional
workflow so that the experiment measures memory/QoS-aware service choice rather
than natural-language decomposition.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


Json = dict[str, Any]

METRICS = [
    "average number of downloads",
    "average number of likes",
    "average response time",
    "average waiting time",
    "reliability",
    "successability",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset/HSC-main")
    parser.add_argument("--output-json", default="outputs/hsc_external_validation.json")
    parser.add_argument("--output-tex", default="outputs/hsc_external_validation.tex")
    parser.add_argument("--train-size", type=int, default=7000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--memory-k", type=int, default=16)
    parser.add_argument(
        "--selection-objective",
        choices=["balanced", "service_qos", "service_first", "qos_first"],
        default="balanced",
        help="Validation objective used to select the GEMS operating point.",
    )
    parser.add_argument(
        "--gems-routing",
        choices=["global", "function"],
        default="function",
        help="Use one validation-selected GEMS operating point globally or select it per HSC function.",
    )
    parser.add_argument(
        "--table-scope",
        choices=["full", "graph_memory"],
        default="full",
        help="Rows included in the exported LaTeX table. JSON always contains all rows.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def flatten_flow(workflow: Json) -> list[str]:
    return [fn for stage in workflow["flow"] for fn in stage]


def service_text(service: Json) -> str:
    return normalize_text(
        " ".join(
            [
                service["service_name"],
                service["function"].replace("/", " "),
                service["function"].replace("-", " "),
            ]
        )
    )


def metric_value(service: Json, metric: str, *, utility: bool) -> float:
    if metric == "average number of downloads":
        value = float(service["normalized_downloads"])
    elif metric == "average number of likes":
        value = float(service["normalized_likes"])
    elif metric == "average response time":
        value = float(service["normalized_response_times"])
        return 1.0 - value if utility else value
    elif metric == "average waiting time":
        value = float(service["normalized_waiting_times"])
        return 1.0 - value if utility else value
    elif metric == "reliability":
        value = float(service["reliabilities"])
    elif metric == "successability":
        value = float(service["successabilities"])
    else:
        value = 0.0
    return value


def service_utility(service: Json, obj_func: Json) -> float:
    return sum(float(weight) * metric_value(service, metric, utility=True) for metric, weight in obj_func.items())


def objective_signature(workflow: Json, *, include_constraints: bool = False) -> tuple[float, ...]:
    obj = tuple(round(float(workflow["obj_func"].get(metric, 0.0)), 3) for metric in METRICS)
    if not include_constraints:
        return obj
    cons = tuple(round(float(workflow["cons"].get(metric, 0.0)), 3) for metric in METRICS)
    return obj + cons


def constraint_ok(service: Json, cons: Json) -> bool:
    for metric, threshold in cons.items():
        threshold = float(threshold)
        if threshold <= 0:
            continue
        value = metric_value(service, metric, utility=False)
        if metric in {"average response time", "average waiting time"}:
            if value >= threshold:
                return False
        else:
            if value <= threshold:
                return False
    return True


def selected_metrics(indices: list[int], models: list[Json], workflow: Json, gold: list[int]) -> Json:
    utilities = [service_utility(models[idx], workflow["obj_func"]) for idx in indices]
    constraints = [constraint_ok(models[idx], workflow["cons"]) for idx in indices]
    rel = [float(models[idx]["reliabilities"]) for idx in indices]
    succ = [float(models[idx]["successabilities"]) for idx in indices]
    return {
        "service_hits": sum(int(a == b) for a, b in zip(indices, gold)),
        "slots": len(gold),
        "workflow_exact": int(indices == gold),
        "utility": float(np.mean(utilities)) if utilities else 0.0,
        "constraint_hits": sum(int(x) for x in constraints),
        "reliable_hits": sum(int(x >= 0.7) for x in rel),
        "success_hits": sum(int(x >= 0.7) for x in succ),
        "reliability": float(np.mean(rel)) if rel else 0.0,
        "successability": float(np.mean(succ)) if succ else 0.0,
    }


def candidate_indices_by_function(models: list[Json]) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, model in enumerate(models):
        buckets[model["function"]].append(idx)
    return buckets


def build_memory(requirements: list[str], workflows: list[Json], solutions: dict[int, Json], train_ids: list[int]) -> list[Json]:
    memory = []
    for req_id in train_ids:
        slots = flatten_flow(workflows[req_id])
        gold = solutions[req_id]["best_solution"]
        for slot_pos, (function, service_idx) in enumerate(zip(slots, gold)):
            memory.append(
                {
                    "req_id": req_id,
                    "slot_pos": slot_pos,
                    "function": function,
                    "service_idx": service_idx,
                    "text": normalize_text(requirements[req_id] + " " + function.replace("/", " ")),
                }
            )
    return memory


def pick_by_score(candidates: list[int], score_fn) -> int:
    return max(candidates, key=lambda idx: (score_fn(idx), -idx))


def evaluate(args: argparse.Namespace) -> Json:
    data_dir = Path(args.data_dir)
    requirements: list[str] = load_json(data_dir / "requirements.json")
    workflows: list[Json] = load_json(data_dir / "workflow.json")
    solutions = {item["req_number"]: item for item in load_json(data_dir / "best_solution.json")}
    models: list[Json] = load_json(data_dir / "normalized_model.json")

    total = min(len(requirements), len(workflows), len(solutions))
    train_end = min(args.train_size, total)
    val_end = min(train_end + args.val_size, total)
    test_end = min(val_end + args.test_size, total)
    train_ids = list(range(0, train_end))
    val_ids = list(range(train_end, val_end))
    test_ids = list(range(val_end, test_end))

    buckets = candidate_indices_by_function(models)
    global_popularity = {
        idx: float(model["normalized_downloads"]) + float(model["normalized_likes"]) for idx, model in enumerate(models)
    }
    train_choice_counts: dict[str, Counter[int]] = defaultdict(Counter)
    train_objective_counts: dict[tuple[str, tuple[float, ...]], Counter[int]] = defaultdict(Counter)
    train_objective_constraint_counts: dict[tuple[str, tuple[float, ...]], Counter[int]] = defaultdict(Counter)
    for req_id in train_ids:
        workflow = workflows[req_id]
        obj_sig = objective_signature(workflow)
        obj_cons_sig = objective_signature(workflow, include_constraints=True)
        for function, service_idx in zip(flatten_flow(workflow), solutions[req_id]["best_solution"]):
            train_choice_counts[function][service_idx] += 1
            train_objective_counts[(function, obj_sig)][service_idx] += 1
            train_objective_constraint_counts[(function, obj_cons_sig)][service_idx] += 1

    memory = build_memory(requirements, workflows, solutions, train_ids)
    memory_texts = [item["text"] for item in memory]
    eval_ids = val_ids + test_ids
    req_texts = [normalize_text(requirements[idx]) for idx in eval_ids]
    vectorizer = TfidfVectorizer(min_df=2, max_features=50000)
    tfidf = vectorizer.fit_transform(memory_texts + req_texts + [service_text(model) for model in models])
    memory_matrix = tfidf[: len(memory)]
    eval_matrix = tfidf[len(memory) : len(memory) + len(req_texts)]
    val_matrix = eval_matrix[: len(val_ids)]
    test_matrix = eval_matrix[len(val_ids) :]
    service_matrix = tfidf[len(memory) + len(req_texts) :]

    memory_by_function: dict[str, list[int]] = defaultdict(list)
    for pos, item in enumerate(memory):
        memory_by_function[item["function"]].append(pos)

    methods = [
        "Popularity",
        "QoS-greedy",
        "Reliability-greedy",
        "Trace-RAG",
        "GraphRAG-static",
        "GEMS",
    ]

    def slot_features(split_ids: list[int], split_matrix) -> list[Json]:
        slots_out: list[Json] = []
        for local_pos, req_id in enumerate(split_ids):
            workflow = workflows[req_id]
            gold = solutions[req_id]["best_solution"]
            req_vec = split_matrix[local_pos]

            for slot_pos, function in enumerate(flatten_flow(workflow)):
                candidates = buckets[function]
                cand_matrix = service_matrix[candidates]
                sem_scores = cosine_similarity(req_vec, cand_matrix).ravel()

                mem_positions = memory_by_function.get(function, [])
                mem_scores_by_service: Counter[int] = Counter()
                if mem_positions:
                    scores = cosine_similarity(req_vec, memory_matrix[mem_positions]).ravel()
                    top_local = np.argsort(-scores)[: args.memory_k]
                    for rank, local_mem_idx in enumerate(top_local):
                        mem_item = memory[mem_positions[int(local_mem_idx)]]
                        mem_scores_by_service[mem_item["service_idx"]] += float(scores[int(local_mem_idx)]) / (rank + 1)

                total_count = max(1, sum(train_choice_counts[function].values()))
                freq = np.asarray([train_choice_counts[function][idx] / total_count for idx in candidates], dtype=float)
                obj_sig = objective_signature(workflow)
                obj_cons_sig = objective_signature(workflow, include_constraints=True)
                obj_total = max(1, sum(train_objective_counts[(function, obj_sig)].values()))
                objcons_total = max(1, sum(train_objective_constraint_counts[(function, obj_cons_sig)].values()))
                obj_mem = np.asarray(
                    [train_objective_counts[(function, obj_sig)][idx] / obj_total for idx in candidates],
                    dtype=float,
                )
                objcons_mem = np.asarray(
                    [train_objective_constraint_counts[(function, obj_cons_sig)][idx] / objcons_total for idx in candidates],
                    dtype=float,
                )
                mem = np.asarray([mem_scores_by_service.get(idx, 0.0) for idx in candidates], dtype=float)
                if mem.size and float(mem.max()) > 0:
                    mem = mem / float(mem.max())
                qos = np.asarray([service_utility(models[idx], workflow["obj_func"]) for idx in candidates], dtype=float)
                cons = np.asarray([1.0 if constraint_ok(models[idx], workflow["cons"]) else 0.0 for idx in candidates], dtype=float)
                rel = np.asarray([float(models[idx]["reliabilities"]) for idx in candidates], dtype=float)
                succ = np.asarray([float(models[idx]["successabilities"]) for idx in candidates], dtype=float)
                pop = np.asarray([global_popularity[idx] for idx in candidates], dtype=float)
                if pop.size and float(pop.max()) > 0:
                    pop = pop / float(pop.max())

                slots_out.append(
                    {
                        "req_id": req_id,
                        "slot_pos": slot_pos,
                        "function": function,
                        "workflow": workflow,
                        "gold": gold[slot_pos],
                        "candidates": candidates,
                        "features": {
                            "sem": np.asarray(sem_scores, dtype=float),
                            "freq": freq,
                            "obj_mem": obj_mem,
                            "objcons_mem": objcons_mem,
                            "mem": mem,
                            "qos": qos,
                            "constraint": cons,
                            "rel": rel,
                            "succ": succ,
                            "pop": pop,
                        },
                    }
                )
        return slots_out

    val_slots = slot_features(val_ids, val_matrix)
    test_slots = slot_features(test_ids, test_matrix)

    def choose(slot: Json, weights: dict[str, float]) -> int:
        score = np.zeros(len(slot["candidates"]), dtype=float)
        for key, weight in weights.items():
            if weight:
                score += float(weight) * slot["features"][key]
        best_pos = int(np.argmax(score))
        return int(slot["candidates"][best_pos])

    def choose_rerank(slot: Json, weights: dict[str, float]) -> int:
        trace_score = 0.65 * slot["features"]["sem"] + 0.35 * slot["features"]["freq"]
        top_k = int(weights.get("top_k", 5))
        top_positions = np.argsort(-trace_score)[: max(1, min(top_k, len(trace_score)))]
        rerank = np.zeros(len(top_positions), dtype=float)
        for key, weight in weights.items():
            if key == "top_k" or not weight:
                continue
            rerank += float(weight) * slot["features"][key][top_positions]
        best_pos = int(top_positions[int(np.argmax(rerank))])
        return int(slot["candidates"][best_pos])

    fixed_weights = {
        "Popularity": {"pop": 1.0},
        "QoS-greedy": {"qos": 1.0},
        "Reliability-greedy": {"rel": 0.6, "succ": 0.4},
        "Trace-RAG": {"sem": 0.65, "freq": 0.35},
        "GraphRAG-static": {"sem": 0.45, "mem": 0.35, "qos": 0.20},
    }

    def build_workflow_choices(slots: list[Json], weights: dict[str, float]) -> dict[int, list[int]]:
        choices: dict[int, list[int]] = defaultdict(list)
        for slot in slots:
            choices[slot["req_id"]].append(choose(slot, weights))
        return choices

    def build_workflow_choices_by_function(
        slots: list[Json],
        function_weights: dict[str, dict[str, float]],
        default_weights: dict[str, float],
    ) -> dict[int, list[int]]:
        choices: dict[int, list[int]] = defaultdict(list)
        for slot in slots:
            function = slot["function"]
            weights = function_weights.get(function, default_weights)
            choices[slot["req_id"]].append(choose(slot, weights))
        return choices

    def build_workflow_choices_by_function_rerank(
        slots: list[Json],
        function_weights: dict[str, dict[str, float]],
        default_weights: dict[str, float],
    ) -> dict[int, list[int]]:
        choices: dict[int, list[int]] = defaultdict(list)
        for slot in slots:
            function = slot["function"]
            weights = function_weights.get(function, default_weights)
            choices[slot["req_id"]].append(choose_rerank(slot, weights))
        return choices

    def summarize(slots: list[Json], weights: dict[str, float]) -> Json:
        choices = build_workflow_choices_by_function_rerank(slots, {}, weights) if "top_k" in weights else build_workflow_choices(slots, weights)
        totals: Counter[str] = Counter()
        float_totals: Counter[str] = Counter()
        by_req: dict[int, list[Json]] = defaultdict(list)
        for slot in slots:
            by_req[slot["req_id"]].append(slot)
        for req_id, req_slots in by_req.items():
            req_slots = sorted(req_slots, key=lambda item: item["slot_pos"])
            chosen = choices[req_id]
            gold = [slot["gold"] for slot in req_slots]
            workflow = req_slots[0]["workflow"]
            metrics = selected_metrics(chosen, models, workflow, gold)
            for key in ["service_hits", "slots", "workflow_exact", "constraint_hits", "reliable_hits", "success_hits"]:
                totals[key] += int(metrics[key])
            for key in ["utility", "reliability", "successability"]:
                float_totals[key] += float(metrics[key])
            totals["workflows"] += 1

        workflows_n = max(1, totals["workflows"])
        slots_n = max(1, totals["slots"])
        row = {
            "service_acc": totals["service_hits"] / slots_n,
            "workflow_exact": totals["workflow_exact"] / workflows_n,
            "qos_score": float_totals["utility"] / workflows_n,
            "constraint_sr": totals["constraint_hits"] / slots_n,
            "reliable_at_1": totals["reliable_hits"] / slots_n,
            "success_at_1": totals["success_hits"] / slots_n,
            "avg_reliability": float_totals["reliability"] / workflows_n,
            "avg_successability": float_totals["successability"] / workflows_n,
            "workflows": totals["workflows"],
            "slots": totals["slots"],
        }
        row["balanced_score"] = float(
            np.mean([row["service_acc"], row["qos_score"], row["constraint_sr"], row["reliable_at_1"], row["success_at_1"]])
        )
        return row

    def candidate_gems_weights() -> list[dict[str, float]]:
        raw_configs = [
            {"sem": 0.00, "freq": 1.00, "mem": 0.00, "qos": 0.00, "constraint": 0.00, "rel": 0.00, "succ": 0.00},
            {"sem": 0.00, "freq": 0.00, "obj_mem": 1.00, "mem": 0.00, "qos": 0.00, "constraint": 0.00, "rel": 0.00, "succ": 0.00},
            {"sem": 0.00, "freq": 0.00, "objcons_mem": 1.00, "mem": 0.00, "qos": 0.00, "constraint": 0.00, "rel": 0.00, "succ": 0.00},
            {"sem": 0.00, "freq": 0.20, "obj_mem": 0.60, "mem": 0.00, "qos": 0.20, "constraint": 0.00, "rel": 0.00, "succ": 0.00},
            {"sem": 0.00, "freq": 0.15, "objcons_mem": 0.65, "mem": 0.00, "qos": 0.20, "constraint": 0.00, "rel": 0.00, "succ": 0.00},
            {"sem": 0.00, "freq": 0.15, "obj_mem": 0.55, "mem": 0.00, "qos": 0.20, "constraint": 0.05, "rel": 0.03, "succ": 0.02},
            {"sem": 0.00, "freq": 0.10, "objcons_mem": 0.60, "mem": 0.00, "qos": 0.20, "constraint": 0.05, "rel": 0.03, "succ": 0.02},
            {"sem": 0.65, "freq": 0.35, "mem": 0.00, "qos": 0.00, "constraint": 0.00, "rel": 0.00, "succ": 0.00},
            {"sem": 0.00, "freq": 0.95, "mem": 0.00, "qos": 0.05, "constraint": 0.00, "rel": 0.00, "succ": 0.00},
            {"sem": 0.00, "freq": 0.90, "mem": 0.00, "qos": 0.10, "constraint": 0.00, "rel": 0.00, "succ": 0.00},
            {"sem": 0.00, "freq": 0.85, "mem": 0.00, "qos": 0.15, "constraint": 0.00, "rel": 0.00, "succ": 0.00},
            {"sem": 0.00, "freq": 0.00, "mem": 0.00, "qos": 1.00, "constraint": 0.00, "rel": 0.00, "succ": 0.00},
            {"sem": 0.00, "freq": 0.00, "mem": 0.00, "qos": 0.90, "constraint": 0.10, "rel": 0.00, "succ": 0.00},
            {"sem": 0.00, "freq": 0.00, "mem": 0.00, "qos": 0.85, "constraint": 0.05, "rel": 0.05, "succ": 0.05},
            {"top_k": 2, "freq": 0.55, "qos": 0.30, "constraint": 0.05, "rel": 0.05, "succ": 0.05},
            {"top_k": 3, "freq": 0.50, "qos": 0.35, "constraint": 0.05, "rel": 0.05, "succ": 0.05},
            {"top_k": 5, "freq": 0.45, "qos": 0.40, "constraint": 0.05, "rel": 0.05, "succ": 0.05},
            {"top_k": 10, "freq": 0.35, "qos": 0.50, "constraint": 0.05, "rel": 0.05, "succ": 0.05},
            {"top_k": 3, "qos": 0.70, "constraint": 0.10, "rel": 0.10, "succ": 0.10},
            {"top_k": 5, "qos": 0.70, "constraint": 0.10, "rel": 0.10, "succ": 0.10},
            {"top_k": 10, "qos": 0.70, "constraint": 0.10, "rel": 0.10, "succ": 0.10},
            {"top_k": 3, "freq": 0.70, "qos": 0.20, "constraint": 0.05, "rel": 0.03, "succ": 0.02},
            {"top_k": 5, "freq": 0.70, "qos": 0.20, "constraint": 0.05, "rel": 0.03, "succ": 0.02},
            {"sem": 0.30, "freq": 0.20, "mem": 0.20, "qos": 0.10, "constraint": 0.05, "rel": 0.10, "succ": 0.05},
            {"sem": 0.20, "freq": 0.40, "mem": 0.15, "qos": 0.05, "constraint": 0.05, "rel": 0.10, "succ": 0.05},
            {"sem": 0.15, "freq": 0.45, "mem": 0.20, "qos": 0.05, "constraint": 0.05, "rel": 0.05, "succ": 0.05},
            {"sem": 0.10, "freq": 0.55, "mem": 0.20, "qos": 0.05, "constraint": 0.05, "rel": 0.03, "succ": 0.02},
            {"sem": 0.15, "freq": 0.25, "mem": 0.20, "qos": 0.20, "constraint": 0.05, "rel": 0.10, "succ": 0.05},
            {"sem": 0.10, "freq": 0.20, "mem": 0.15, "qos": 0.30, "constraint": 0.10, "rel": 0.10, "succ": 0.05},
            {"sem": 0.10, "freq": 0.15, "mem": 0.15, "qos": 0.20, "constraint": 0.10, "rel": 0.20, "succ": 0.10},
            {"sem": 0.05, "freq": 0.15, "mem": 0.10, "qos": 0.15, "constraint": 0.10, "rel": 0.30, "succ": 0.15},
            {"sem": 0.05, "freq": 0.10, "mem": 0.10, "qos": 0.10, "constraint": 0.10, "rel": 0.35, "succ": 0.20},
            {"sem": 0.20, "freq": 0.30, "mem": 0.20, "qos": 0.10, "constraint": 0.05, "rel": 0.10, "succ": 0.05},
            {"sem": 0.10, "freq": 0.35, "mem": 0.25, "qos": 0.10, "constraint": 0.05, "rel": 0.10, "succ": 0.05},
            {"sem": 0.10, "freq": 0.25, "mem": 0.35, "qos": 0.10, "constraint": 0.05, "rel": 0.10, "succ": 0.05},
            {"sem": 0.05, "freq": 0.45, "mem": 0.25, "qos": 0.10, "constraint": 0.05, "rel": 0.07, "succ": 0.03},
            {"sem": 0.05, "freq": 0.35, "mem": 0.35, "qos": 0.10, "constraint": 0.05, "rel": 0.07, "succ": 0.03},
            {"sem": 0.02, "freq": 0.78, "mem": 0.10, "qos": 0.05, "constraint": 0.02, "rel": 0.02, "succ": 0.01},
            {"sem": 0.02, "freq": 0.68, "mem": 0.10, "qos": 0.15, "constraint": 0.02, "rel": 0.02, "succ": 0.01},
            {"sem": 0.02, "freq": 0.58, "mem": 0.10, "qos": 0.25, "constraint": 0.02, "rel": 0.02, "succ": 0.01},
            {"sem": 0.02, "freq": 0.48, "mem": 0.10, "qos": 0.35, "constraint": 0.02, "rel": 0.02, "succ": 0.01},
            {"sem": 0.02, "freq": 0.38, "mem": 0.10, "qos": 0.45, "constraint": 0.02, "rel": 0.02, "succ": 0.01},
            {"sem": 0.02, "freq": 0.28, "mem": 0.10, "qos": 0.55, "constraint": 0.02, "rel": 0.02, "succ": 0.01},
            {"sem": 0.02, "freq": 0.18, "mem": 0.10, "qos": 0.65, "constraint": 0.02, "rel": 0.02, "succ": 0.01},
            {"sem": 0.02, "freq": 0.08, "mem": 0.10, "qos": 0.75, "constraint": 0.02, "rel": 0.02, "succ": 0.01},
            {"sem": 0.00, "freq": 0.85, "mem": 0.10, "qos": 0.05, "constraint": 0.00, "rel": 0.00, "succ": 0.00},
            {"sem": 0.00, "freq": 0.65, "mem": 0.10, "qos": 0.25, "constraint": 0.00, "rel": 0.00, "succ": 0.00},
            {"sem": 0.00, "freq": 0.45, "mem": 0.10, "qos": 0.45, "constraint": 0.00, "rel": 0.00, "succ": 0.00},
            {"sem": 0.00, "freq": 0.25, "mem": 0.10, "qos": 0.65, "constraint": 0.00, "rel": 0.00, "succ": 0.00},
            {"sem": 0.00, "freq": 0.05, "mem": 0.10, "qos": 0.85, "constraint": 0.00, "rel": 0.00, "succ": 0.00},
        ]
        configs = []
        for raw in raw_configs:
            total_weight = sum(value for key, value in raw.items() if key != "top_k")
            normalized = {key: value / total_weight for key, value in raw.items() if key != "top_k"}
            if "top_k" in raw:
                normalized["top_k"] = raw["top_k"]
            configs.append(normalized)
        return configs

    best_gems_weights: dict[str, float] | None = None
    best_val: Json | None = None
    def selection_key(row: Json) -> tuple[float, ...]:
        if args.selection_objective == "service_qos":
            return (
                0.45 * row["service_acc"] + 0.45 * row["qos_score"] + 0.10 * row["balanced_score"],
                row["service_acc"],
                row["qos_score"],
            )
        if args.selection_objective == "service_first":
            return (row["service_acc"], row["qos_score"], row["balanced_score"])
        if args.selection_objective == "qos_first":
            return (row["qos_score"], row["service_acc"], row["balanced_score"])
        return (row["balanced_score"], row["service_acc"], row["qos_score"])

    for weights in candidate_gems_weights():
        val_row = summarize(val_slots, weights)
        if best_val is None or selection_key(val_row) > selection_key(best_val):
            best_val = val_row
            best_gems_weights = weights

    assert best_gems_weights is not None and best_val is not None

    function_weights: dict[str, dict[str, float]] = {}
    if args.gems_routing == "function":
        val_by_function: dict[str, list[Json]] = defaultdict(list)
        for slot in val_slots:
            val_by_function[slot["function"]].append(slot)
        for function, function_slots in val_by_function.items():
            if len(function_slots) < 3:
                continue
            local_best: Json | None = None
            local_weights: dict[str, float] | None = None
            for weights in candidate_gems_weights():
                row = summarize(function_slots, weights)
                if local_best is None or selection_key(row) > selection_key(local_best):
                    local_best = row
                    local_weights = weights
            if local_weights is not None:
                function_weights[function] = local_weights

    method_totals: dict[str, Counter[str]] = defaultdict(Counter)
    method_float_totals: dict[str, Counter[str]] = defaultdict(Counter)

    def add_result(method: str, chosen: list[int], workflow: Json, gold: list[int]) -> None:
        metrics = selected_metrics(chosen, models, workflow, gold)
        for key in ["service_hits", "slots", "workflow_exact", "constraint_hits", "reliable_hits", "success_hits"]:
            method_totals[method][key] += int(metrics[key])
        for key in ["utility", "reliability", "successability"]:
            method_float_totals[method][key] += float(metrics[key])
        method_totals[method]["workflows"] += 1

    for method in methods:
        weights = best_gems_weights if method == "GEMS" else fixed_weights[method]
        if method == "GEMS" and args.gems_routing == "function":
            if any("top_k" in weights for weights in function_weights.values()) or "top_k" in best_gems_weights:
                choices = build_workflow_choices_by_function_rerank(test_slots, function_weights, best_gems_weights)
            else:
                choices = build_workflow_choices_by_function(test_slots, function_weights, best_gems_weights)
        else:
            choices = build_workflow_choices_by_function_rerank(test_slots, {}, weights) if "top_k" in weights else build_workflow_choices(test_slots, weights)
        by_req: dict[int, list[Json]] = defaultdict(list)
        for slot in test_slots:
            by_req[slot["req_id"]].append(slot)
        for req_id, req_slots in by_req.items():
            req_slots = sorted(req_slots, key=lambda item: item["slot_pos"])
            workflow = req_slots[0]["workflow"]
            gold = [slot["gold"] for slot in req_slots]
            add_result(method, choices[req_id], workflow, gold)

    rows = []
    for method in methods:
        totals = method_totals[method]
        floats = method_float_totals[method]
        workflows_n = max(1, totals["workflows"])
        slots_n = max(1, totals["slots"])
        rows.append(
            {
                "method": method,
                "service_acc": totals["service_hits"] / slots_n,
                "workflow_exact": totals["workflow_exact"] / workflows_n,
                "qos_score": floats["utility"] / workflows_n,
                "constraint_sr": totals["constraint_hits"] / slots_n,
                "reliable_at_1": totals["reliable_hits"] / slots_n,
                "success_at_1": totals["success_hits"] / slots_n,
                "avg_reliability": floats["reliability"] / workflows_n,
                "avg_successability": floats["successability"] / workflows_n,
                "workflows": totals["workflows"],
                "slots": totals["slots"],
            }
        )
        rows[-1]["balanced_score"] = float(
            np.mean(
                [
                    rows[-1]["service_acc"],
                    rows[-1]["qos_score"],
                    rows[-1]["constraint_sr"],
                    rows[-1]["reliable_at_1"],
                    rows[-1]["success_at_1"],
                ]
            )
        )

    return {
        "dataset": "HSC-main",
        "split": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "models": len(models),
        "memory_items": len(memory),
        "selection_objective": args.selection_objective,
        "gems_routing": args.gems_routing,
        "gems_validation_metrics": best_val,
        "gems_weights": best_gems_weights,
        "function_specific_weight_count": len(function_weights),
        "metrics": {
            "service_acc": "Selected service equals optimizer-provided best_solution for a gold function slot.",
            "workflow_exact": "All selected services equal the best_solution for a workflow.",
            "qos_score": "Mean objective utility of selected services under HSC objective weights.",
            "constraint_sr": "Fraction of selected services satisfying nonzero HSC constraints.",
            "reliable_at_1": "Fraction of selected services with reliability >= 0.7.",
            "success_at_1": "Fraction of selected services with successability >= 0.7.",
            "balanced_score": "Mean of Service.Acc, QoS.Score, Constraint.SR, Reliable@1, and Success@1.",
        },
        "rows": rows,
    }


def bold(value: str, is_best: bool) -> str:
    return f"\\textbf{{{value}}}" if is_best else value


def write_tex(result: Json, path: Path, table_scope: str) -> None:
    rows = result["rows"]
    if table_scope == "graph_memory":
        rows = [row for row in rows if row["method"] in {"GraphRAG-static", "GEMS"}]
    higher = ["service_acc", "qos_score", "constraint_sr", "reliable_at_1", "success_at_1", "balanced_score"]
    best = {metric: max(row[metric] for row in rows) for metric in higher}
    caption = (
        "Graph-memory external validation on the HSC AI service composition benchmark."
        if table_scope == "graph_memory"
        else "External validation on the HSC AI service composition benchmark."
    )
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:hsc-external-validation}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Method & Service.Acc $\\uparrow$ & QoS.Score $\\uparrow$ & Constraint.SR $\\uparrow$ & Reliable@1 $\\uparrow$ & Success@1 $\\uparrow$ & Balanced $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        vals = []
        for metric in higher:
            text = f"{row[metric]:.4f}"
            vals.append(bold(text, math.isclose(row[metric], best[metric], rel_tol=1e-12, abs_tol=1e-12)))
        method = "\\textsc{GEMS}" if row["method"] == "GEMS" else row["method"]
        lines.append(f"{method} & " + " & ".join(vals) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    output_json = Path(args.output_json)
    output_tex = Path(args.output_tex)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2))
    write_tex(result, output_tex, args.table_scope)
    print(json.dumps({"output_json": str(output_json), "output_tex": str(output_tex), "rows": result["rows"]}, indent=2))


if __name__ == "__main__":
    main()
