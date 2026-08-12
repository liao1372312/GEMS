#!/usr/bin/env python
"""Proxy hazard, online-batch, and error analysis for public composition.

The public composition file does not contain real temporal API evolution or
live repair logs. This script therefore constructs deterministic proxy slices
from available candidate metadata and evaluates the same local scorers used by
the main public-composition experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gems.graph_memory import ExecutionGraphMemory
from run_public_composition_experiments import (
    ExperimentScorer,
    StepExample,
    TraceRagScorer,
    add_composition_traces_to_memory,
    candidate_text,
    endpoint_key,
    ensure_candidate_nodes,
    f1_score,
    load_composition_examples,
    load_processed_endpoint_maps,
    rank_from_scores,
    required_param_names,
    split_records,
)


Json = dict[str, Any]


@dataclass
class Prediction:
    record_id: str
    step_id: int
    method: str
    pred_index: int
    gold_index: int
    correct: bool
    pred_api: str
    gold_api: str
    pred_domain: str
    gold_domain: str
    pred_params: list[str]
    gold_params: list[str]
    pred_observed_success: bool | None
    semantic_top1_observed_success: bool | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--processed-endpoints", default="dataset/processed/endpoints.jsonl")
    parser.add_argument("--memory", default="outputs/gems_memory_train_only.json")
    parser.add_argument("--output", default="outputs/public_hazard_online_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--trace-top-k", type=int, default=16)
    parser.add_argument("--batches", type=int, default=5)
    return parser.parse_args()


def setup(args: argparse.Namespace) -> tuple[list[Json], list[StepExample], dict[str, set[str]], ExperimentScorer, dict[str, Json]]:
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    train_examples = [example for example in examples if example.record_id in splits["train"]]
    endpoints_by_url, _ = load_processed_endpoint_maps(args.processed_endpoints)
    memory = ExecutionGraphMemory.load(args.memory) if Path(args.memory).exists() else ExecutionGraphMemory()
    key_to_node_id = ensure_candidate_nodes(memory, examples, endpoints_by_url)
    add_composition_traces_to_memory(memory, train_examples, key_to_node_id)
    memory.propagate_reliability(layers=2)
    scorer = ExperimentScorer(
        memory=memory,
        trace_scorer=TraceRagScorer(train_examples),
        key_to_node_id=key_to_node_id,
        endpoints_by_url=endpoints_by_url,
        trace_top_k=args.trace_top_k,
    )
    return records, examples, splits, scorer, endpoints_by_url


def observed_success(candidate: Json, endpoints_by_url: dict[str, Json]) -> bool | None:
    endpoint = endpoints_by_url.get(candidate.get("endpoint"))
    if not endpoint:
        return None
    value = endpoint.get("observed_success")
    return value if isinstance(value, bool) else None


def domain_token(value: str) -> set[str]:
    return {part.lower() for part in value.replace("/", " ").replace("-", " ").split() if len(part) > 2}


def candidate_param_f1(candidate: Json, gold: Json) -> float:
    return f1_score(required_param_names(candidate), required_param_names(gold))


def same_domain(candidate: Json, example: StepExample) -> bool:
    return bool(domain_token(str(candidate.get("domain") or "")) & domain_token(example.record_domain))


def predict(method: str, example: StepExample, scorer: ExperimentScorer, endpoints_by_url: dict[str, Json]) -> Prediction:
    scores, _ = scorer.score(method, example)
    ranked = rank_from_scores(scores)
    pred_index = ranked[0] if ranked else 0
    pred = example.candidates[pred_index]
    gold = example.gold_candidate
    return Prediction(
        record_id=example.record_id,
        step_id=example.step_id,
        method=method,
        pred_index=pred_index,
        gold_index=example.gold_index,
        correct=pred_index == example.gold_index,
        pred_api=str(pred.get("api_name") or ""),
        gold_api=str(gold.get("api_name") or ""),
        pred_domain=str(pred.get("domain") or ""),
        gold_domain=str(gold.get("domain") or ""),
        pred_params=sorted(required_param_names(pred)),
        gold_params=sorted(required_param_names(gold)),
        pred_observed_success=observed_success(pred, endpoints_by_url),
        semantic_top1_observed_success=observed_success(example.candidates[0], endpoints_by_url),
    )


def score_all_methods(methods: list[str], example: StepExample, scorer: ExperimentScorer) -> dict[str, list[float]]:
    trace_scores, retrieved = scorer.trace_scorer.candidate_scores(example, top_k=scorer.trace_top_k)
    candidate_node_ids = [
        scorer.key_to_node_id.get(endpoint_key(candidate)) or scorer.key_to_node_id.get(str(candidate.get("api_name") or ""))
        for candidate in example.candidates
    ]
    valid_node_ids = [node_id for node_id in candidate_node_ids if node_id in scorer.memory.nodes]
    graph_scores = scorer.retriever.score_nodes(
        " ".join([example.user_query, example.step_description, example.record_domain, " ".join(example.required_inputs)]),
        "provider",
        valid_node_ids,
    ) if valid_node_ids else {}

    out = {method: [] for method in methods}
    for index, candidate in enumerate(example.candidates):
        sim = float(candidate.get("similarity_score") or 0.0)
        trace = max(
            trace_scores.get(endpoint_key(candidate), 0.0),
            trace_scores.get(str(candidate.get("api_name") or ""), 0.0),
        )
        node_id = candidate_node_ids[index]
        node = scorer.memory.nodes.get(node_id or "")
        reliability = float(node.reliability) if node else 0.5
        risk = float(node.risk) if node else 0.0
        graph = float(graph_scores.get(node_id or "", 0.0))
        endpoint = scorer.endpoints_by_url.get(candidate.get("endpoint"))
        observed = endpoint.get("observed_success") if endpoint else None
        success_bonus = 0.0 if observed is None else (0.08 if observed is True else -0.08)

        param_overlap = 0.0
        params = required_param_names(candidate)
        if params and retrieved:
            names = {trace_item.api_name for trace_item in retrieved[: scorer.trace_top_k]}
            param_overlap = 1.0 if candidate.get("api_name") in names else min(0.5, len(params) / 4.0)

        for method in methods:
            if method == "semantic_top1":
                score = sim
            elif method == "trace_rag":
                score = 0.55 * trace + 0.45 * sim
            elif method == "structmem_rag":
                score = 0.65 * trace + 0.30 * sim + 0.05 * param_overlap
            elif method == "graphrag_static":
                score = 0.50 * sim + 0.30 * graph + 0.20 * trace
            elif method == "gems_no_reliability":
                score = 0.50 * sim + 0.35 * trace + 0.15 * graph
            elif method == "gems_reliability_only":
                score = reliability - 0.25 * risk + success_bonus
            elif method == "gems":
                score = 0.40 * sim + 0.25 * trace + 0.20 * graph + 0.15 * reliability - 0.10 * risk
            elif method == "gems_oracle_gate":
                score = 0.45 * sim + 0.25 * trace + 0.15 * graph + 0.15 * reliability - 0.20 * risk + success_bonus
            else:
                raise ValueError(f"Unknown method: {method}")
            out[method].append(float(score))
    return out


def prediction_from_scores(
    method: str,
    example: StepExample,
    scores: list[float],
    endpoints_by_url: dict[str, Json],
) -> Prediction:
    ranked = rank_from_scores(scores)
    pred_index = ranked[0] if ranked else 0
    pred = example.candidates[pred_index]
    gold = example.gold_candidate
    return Prediction(
        record_id=example.record_id,
        step_id=example.step_id,
        method=method,
        pred_index=pred_index,
        gold_index=example.gold_index,
        correct=pred_index == example.gold_index,
        pred_api=str(pred.get("api_name") or ""),
        gold_api=str(gold.get("api_name") or ""),
        pred_domain=str(pred.get("domain") or ""),
        gold_domain=str(gold.get("domain") or ""),
        pred_params=sorted(required_param_names(pred)),
        gold_params=sorted(required_param_names(gold)),
        pred_observed_success=observed_success(pred, endpoints_by_url),
        semantic_top1_observed_success=observed_success(example.candidates[0], endpoints_by_url),
    )


def prediction_key(method: str, example: StepExample) -> tuple[str, str, int]:
    return method, example.record_id, example.step_id


def build_prediction_cache(
    methods: list[str],
    examples: list[StepExample],
    scorer: ExperimentScorer,
    endpoints_by_url: dict[str, Json],
) -> dict[tuple[str, str, int], Prediction]:
    cache: dict[tuple[str, str, int], Prediction] = {}
    for example in examples:
        scores_by_method = score_all_methods(methods, example, scorer)
        for method, scores in scores_by_method.items():
            cache[prediction_key(method, example)] = prediction_from_scores(method, example, scores, endpoints_by_url)
    return cache


def hazard_predicates(endpoints_by_url: dict[str, Json]) -> dict[str, Callable[[StepExample], bool]]:
    def deprecated_api(example: StepExample) -> bool:
        top = observed_success(example.candidates[0], endpoints_by_url)
        return top is False and any(observed_success(c, endpoints_by_url) is True for c in example.candidates[1:])

    def schema_drift(example: StepExample) -> bool:
        top_f1 = candidate_param_f1(example.candidates[0], example.gold_candidate)
        return top_f1 < 1.0 and any(candidate_param_f1(c, example.gold_candidate) > top_f1 for c in example.candidates[1:])

    def qos_drift(example: StepExample) -> bool:
        failures = [observed_success(c, endpoints_by_url) is False for c in example.candidates]
        return sum(failures) >= 2 and any(observed_success(c, endpoints_by_url) is True for c in example.candidates)

    def conflicting_memory(example: StepExample) -> bool:
        # Proxy: semantic top-1 is not gold but is very close to gold in score.
        if len(example.candidates) < 2 or example.gold_index == 0:
            return False
        top_sim = float(example.candidates[0].get("similarity_score") or 0.0)
        gold_sim = float(example.gold_candidate.get("similarity_score") or 0.0)
        return abs(top_sim - gold_sim) <= 0.05

    def noisy_failure_trace(example: StepExample) -> bool:
        known = [observed_success(c, endpoints_by_url) for c in example.candidates]
        return known.count(False) >= 1 and known.count(None) >= 3

    def role_mismatched_memory(example: StepExample) -> bool:
        return (not same_domain(example.candidates[0], example)) and any(same_domain(c, example) for c in example.candidates[1:])

    return {
        "Deprecated API": deprecated_api,
        "Schema drift": schema_drift,
        "QoS drift": qos_drift,
        "Conflicting memory": conflicting_memory,
        "Noisy failure trace": noisy_failure_trace,
        "Role-mismatched memory": role_mismatched_memory,
    }


def evaluate_slice(
    methods: list[str],
    examples: list[StepExample],
    scorer: ExperimentScorer,
    endpoints_by_url: dict[str, Json],
    prediction_cache: dict[tuple[str, str, int], Prediction],
) -> dict[str, Json]:
    rows: dict[str, Json] = {}
    for method in methods:
        if not examples:
            rows[method] = {"count": 0, "api_acc": 0.0, "workflow_exact": 0.0, "para_f1": 0.0, "safe_top1_rate": 0.0}
            continue
        preds = [prediction_cache[prediction_key(method, example)] for example in examples]
        known = [p.pred_observed_success for p in preds if p.pred_observed_success is not None]
        by_record: dict[str, list[bool]] = defaultdict(list)
        para_values = []
        for pred in preds:
            by_record[pred.record_id].append(pred.correct)
            para_values.append(f1_score(set(pred.pred_params), set(pred.gold_params)))
        rows[method] = {
            "count": len(examples),
            "api_acc": sum(1 for p in preds if p.correct) / len(preds) if preds else 0.0,
            "workflow_exact": sum(1 for values in by_record.values() if values and all(values)) / len(by_record) if by_record else 0.0,
            "para_f1": sum(para_values) / len(para_values) if para_values else 0.0,
            "safe_top1_rate": sum(1 for value in known if value is True) / len(known) if known else None,
        }
    return rows


def evaluate_hazards(
    methods: list[str],
    test_examples: list[StepExample],
    scorer: ExperimentScorer,
    endpoints_by_url: dict[str, Json],
    prediction_cache: dict[tuple[str, str, int], Prediction],
) -> dict[str, Json]:
    results: dict[str, Json] = {}
    for name, predicate in hazard_predicates(endpoints_by_url).items():
        subset = [example for example in test_examples if predicate(example)]
        results[name] = evaluate_slice(methods, subset, scorer, endpoints_by_url, prediction_cache)
    return results


def evaluate_online(
    methods: list[str],
    test_examples: list[StepExample],
    scorer: ExperimentScorer,
    endpoints_by_url: dict[str, Json],
    batches: int,
    prediction_cache: dict[tuple[str, str, int], Prediction],
) -> dict[str, Json]:
    ordered_records = sorted({example.record_id for example in test_examples})
    chunks = [set(ordered_records[i::batches]) for i in range(batches)]
    by_record: dict[str, list[StepExample]] = defaultdict(list)
    for example in test_examples:
        by_record[example.record_id].append(example)

    out: dict[str, Json] = {}
    prev_failures: dict[str, set[str]] = {method: set() for method in methods}
    for batch_idx, record_ids in enumerate(chunks, start=1):
        batch_examples = [example for rid in record_ids for example in by_record[rid]]
        out[f"T{batch_idx}"] = {}
        for method in methods:
            preds = [prediction_cache[prediction_key(method, example)] for example in batch_examples]
            pred_by_record: dict[str, list[bool]] = defaultdict(list)
            para_values = []
            for pred in preds:
                pred_by_record[pred.record_id].append(pred.correct)
                para_values.append(f1_score(set(pred.pred_params), set(pred.gold_params)))
            failed_records = {p.record_id for p in preds if not p.correct}
            repeated = len(failed_records & prev_failures[method])
            out[f"T{batch_idx}"][method] = {
                "records": len(record_ids),
                "steps": len(batch_examples),
                "api_acc": sum(1 for p in preds if p.correct) / len(preds) if preds else 0.0,
                "workflow_exact": sum(1 for values in pred_by_record.values() if values and all(values)) / len(pred_by_record) if pred_by_record else 0.0,
                "para_f1": sum(para_values) / len(para_values) if para_values else 0.0,
                "repeated_failure_rate": repeated / len(failed_records) if failed_records else 0.0,
            }
            prev_failures[method] |= failed_records
    return out


def error_type(pred: Prediction) -> str:
    if pred.correct:
        return "Correct"
    if pred.pred_observed_success is False:
        return "Stale or failed endpoint reuse"
    if pred.pred_domain and pred.gold_domain and pred.pred_domain != pred.gold_domain:
        return "Wrong domain/API selection"
    if set(pred.pred_params) != set(pred.gold_params):
        return "Parameter binding mismatch"
    if pred.pred_index != pred.gold_index:
        return "Wrong API selection"
    return "Other"


def evaluate_errors(
    methods: list[str],
    test_examples: list[StepExample],
    prediction_cache: dict[tuple[str, str, int], Prediction],
    limit: int = 0,
) -> dict[str, Json]:
    out: dict[str, Json] = {}
    for method in methods:
        preds = [prediction_cache[prediction_key(method, example)] for example in test_examples]
        failures = [pred for pred in preds if not pred.correct]
        if limit:
            failures = failures[:limit]
        counts = Counter(error_type(pred) for pred in failures)
        out[method] = {
            "failures": len(failures),
            "error_counts": dict(counts),
            "examples": [asdict(pred) for pred in failures[:10]],
        }
    return out


def select_case_study(
    test_examples: list[StepExample],
    scorer: ExperimentScorer,
    endpoints_by_url: dict[str, Json],
    prediction_cache: dict[tuple[str, str, int], Prediction],
) -> Json:
    for example in test_examples:
        sem = prediction_cache[prediction_key("semantic_top1", example)]
        gems = prediction_cache[prediction_key("gems", example)]
        llm_candidate = None
        if (not sem.correct) and (gems.pred_observed_success is True or gems.correct):
            llm_candidate = gems
        if llm_candidate:
            return {
                "record_id": example.record_id,
                "step_id": example.step_id,
                "user_query": example.user_query,
                "step_description": example.step_description,
                "semantic_top1": asdict(sem),
                "gems": asdict(gems),
                "gold": {
                    "api": example.gold_candidate.get("api_name"),
                    "domain": example.gold_candidate.get("domain"),
                    "endpoint": example.gold_candidate.get("endpoint"),
                },
                "candidates": [
                    {
                        "rank": idx + 1,
                        "api_name": c.get("api_name"),
                        "domain": c.get("domain"),
                        "similarity_score": c.get("similarity_score"),
                        "observed_success": observed_success(c, endpoints_by_url),
                    }
                    for idx, c in enumerate(example.candidates)
                ],
            }
    example = test_examples[0]
    return {
        "record_id": example.record_id,
        "step_id": example.step_id,
        "user_query": example.user_query,
        "step_description": example.step_description,
        "semantic_top1": asdict(prediction_cache[prediction_key("semantic_top1", example)]),
        "gems": asdict(prediction_cache[prediction_key("gems", example)]),
        "gold": {
            "api": example.gold_candidate.get("api_name"),
            "domain": example.gold_candidate.get("domain"),
            "endpoint": example.gold_candidate.get("endpoint"),
        },
    }


def main() -> None:
    args = parse_args()
    records, examples, splits, scorer, endpoints_by_url = setup(args)
    test_examples = [example for example in examples if example.record_id in splits["test"]]
    methods = ["semantic_top1", "trace_rag", "graphrag_static", "gems_no_reliability", "gems", "gems_reliability_only"]
    prediction_cache = build_prediction_cache(methods, test_examples, scorer, endpoints_by_url)
    payload = {
        "config": vars(args),
        "dataset": {
            "test_records": len(splits["test"]),
            "test_steps": len(test_examples),
            "hazard_definitions": {
                "Deprecated API": "semantic top-1 has failed feedback and another candidate has successful feedback",
                "Schema drift": "semantic top-1 parameter set mismatches gold while another candidate is closer",
                "QoS drift": "candidate list contains multiple failed-feedback endpoints and at least one successful endpoint",
                "Conflicting memory": "semantic top-1 is not gold but has near-tied similarity with gold",
                "Noisy failure trace": "candidate list mixes failed-feedback and unknown-feedback endpoints",
                "Role-mismatched memory": "semantic top-1 domain mismatches request domain while another candidate matches",
            },
        },
        "hazard_results": evaluate_hazards(methods, test_examples, scorer, endpoints_by_url, prediction_cache),
        "online_batches": evaluate_online(methods, test_examples, scorer, endpoints_by_url, args.batches, prediction_cache),
        "error_analysis": evaluate_errors(methods, test_examples, prediction_cache),
        "case_study": select_case_study(test_examples, scorer, endpoints_by_url, prediction_cache),
        "notes": [
            "Hazard and online results are proxy evaluations because the public dataset lacks real temporal API-evolution logs.",
            "Repeated failure is measured over deterministic test batches, not over an adaptive system that modifies its policy online.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Hazard proxy results")
    for hazard, rows in payload["hazard_results"].items():
        print(f"\n{hazard}")
        for method, metrics in rows.items():
            print(
                f"{method:22s} n={metrics['count']:4d} "
                f"api={metrics['api_acc']:.4f} wf={metrics['workflow_exact']:.4f} "
                f"para={metrics['para_f1']:.4f} safe={metrics['safe_top1_rate']}"
            )
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
