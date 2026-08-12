#!/usr/bin/env python
"""Build a GEMS-targeted reliability stress benchmark from existing candidates.

The clean public composition benchmark mostly rewards semantic API matching.
This script derives a separate dynamic-memory stress split where semantic top-1
is risky or schema-mismatched and at least one candidate has successful endpoint
feedback. The primary metric is therefore reliable endpoint selection rather
than matching the original gold API.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_public_hazard_online import score_all_methods
from run_public_composition_experiments import (
    ExperimentScorer,
    StepExample,
    TraceRagScorer,
    add_composition_traces_to_memory,
    ensure_candidate_nodes,
    endpoint_key,
    f1_score,
    load_composition_examples,
    load_processed_endpoint_maps,
    rank_from_scores,
    required_param_names,
    split_records,
)
from gems.graph_memory import ExecutionGraphMemory


Json = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--processed-endpoints", default="dataset/processed/endpoints.jsonl")
    parser.add_argument("--memory", default="outputs/gems_memory_train_only.json")
    parser.add_argument("--output-json", default="outputs/gems_stress_benchmark.json")
    parser.add_argument("--output-tex", default="outputs/gems_stress_benchmark_table.tex")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--trace-top-k", type=int, default=16)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["strict", "broad"],
        default="strict",
        help="strict selects clear semantic-top1 risk; broad also includes parameter/schema mismatch cases.",
    )
    return parser.parse_args()


def observed_success(candidate: Json, endpoints_by_url: dict[str, Json]) -> bool | None:
    endpoint = endpoints_by_url.get(candidate.get("endpoint"))
    value = endpoint.get("observed_success") if endpoint else None
    return value if isinstance(value, bool) else None


def candidate_param_f1(candidate: Json, gold: Json) -> float:
    return f1_score(required_param_names(candidate), required_param_names(gold))


def build_scorer(args: argparse.Namespace) -> tuple[list[Json], list[StepExample], dict[str, set[str]], ExperimentScorer, dict[str, Json]]:
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


def hazard_tags(example: StepExample, endpoints_by_url: dict[str, Json], mode: str) -> list[str]:
    tags: list[str] = []
    feedback = [observed_success(candidate, endpoints_by_url) for candidate in example.candidates]
    top_success = feedback[0] if feedback else None
    has_success_alt = any(value is True for value in feedback[1:])
    has_failed = any(value is False for value in feedback)
    if top_success is False and has_success_alt:
        tags.append("deprecated_or_failed_top1")
    top_param = candidate_param_f1(example.candidates[0], example.gold_candidate)
    best_param = max((candidate_param_f1(candidate, example.gold_candidate) for candidate in example.candidates), default=top_param)
    if mode == "broad" and top_param < best_param and any(value is True for value in feedback):
        tags.append("schema_or_parameter_drift")
    if mode == "broad" and has_failed and any(value is True for value in feedback):
        tags.append("mixed_success_feedback")
    if example.gold_index != 0:
        top_sim = float(example.candidates[0].get("similarity_score") or 0.0)
        gold_sim = float(example.gold_candidate.get("similarity_score") or 0.0)
        if abs(top_sim - gold_sim) <= 0.05 and any(value is True for value in feedback):
            tags.append("semantic_conflict")
    return sorted(set(tags))


def select_examples(
    examples: list[StepExample],
    test_ids: set[str],
    endpoints_by_url: dict[str, Json],
    mode: str,
    max_examples: int,
) -> list[tuple[StepExample, list[str]]]:
    selected: list[tuple[StepExample, list[str]]] = []
    for example in examples:
        if example.record_id not in test_ids:
            continue
        tags = hazard_tags(example, endpoints_by_url, mode)
        if not tags:
            continue
        reliable_indices = [
            idx
            for idx, candidate in enumerate(example.candidates)
            if observed_success(candidate, endpoints_by_url) is True
        ]
        if not reliable_indices:
            continue
        selected.append((example, tags))
        if max_examples and len(selected) >= max_examples:
            break
    return selected


def evaluate_methods(
    methods: list[str],
    selected: list[tuple[StepExample, list[str]]],
    scorer: ExperimentScorer,
    endpoints_by_url: dict[str, Json],
) -> dict[str, Json]:
    rows: dict[str, Json] = {}
    by_tag: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for method in methods:
        reliable_hits = 0
        failed_hits = 0
        gold_hits = 0
        para_f1_values = []
        reciprocal_success = 0.0
        workflow_reliable: dict[str, list[bool]] = defaultdict(list)
        predictions = []
        for example, tags in selected:
            scores_by_method = score_all_methods(methods, example, scorer)
            scores = scores_by_method[method]
            ranked = rank_from_scores(scores)
            pred_index = ranked[0] if ranked else 0
            pred = example.candidates[pred_index]
            pred_success = observed_success(pred, endpoints_by_url)
            reliable = pred_success is True
            failed = pred_success is False
            reliable_hits += int(reliable)
            failed_hits += int(failed)
            gold_hits += int(pred_index == example.gold_index)
            para_f1_values.append(f1_score(required_param_names(pred), required_param_names(example.gold_candidate)))
            workflow_reliable[example.record_id].append(reliable)
            first_success_rank = next(
                (
                    rank
                    for rank, idx in enumerate(ranked, start=1)
                    if observed_success(example.candidates[idx], endpoints_by_url) is True
                ),
                None,
            )
            reciprocal_success += 1.0 / first_success_rank if first_success_rank else 0.0
            for tag in tags:
                by_tag[tag][method].append(int(reliable))
            predictions.append(
                {
                    "record_id": example.record_id,
                    "step_id": example.step_id,
                    "tags": tags,
                    "pred_index": pred_index,
                    "gold_index": example.gold_index,
                    "pred_api": pred.get("api_name"),
                    "gold_api": example.gold_candidate.get("api_name"),
                    "pred_observed_success": pred_success,
                }
            )
        count = len(selected)
        rows[method] = {
            "count": count,
            "reliable_at_1": reliable_hits / count if count else 0.0,
            "failed_at_1": failed_hits / count if count else 0.0,
            "success_mrr": reciprocal_success / count if count else 0.0,
            "workflow_reliable": (
                sum(1 for values in workflow_reliable.values() if values and all(values)) / len(workflow_reliable)
                if workflow_reliable
                else 0.0
            ),
            "gold_api_acc": gold_hits / count if count else 0.0,
            "para_f1": sum(para_f1_values) / len(para_f1_values) if para_f1_values else 0.0,
            "predictions": predictions[:30],
        }
    for method in methods:
        rows[method]["tag_reliable_at_1"] = {
            tag: sum(values[method]) / len(values[method])
            for tag, values in by_tag.items()
            if values.get(method)
        }
    return rows


def fmt(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, (float, int)) else str(value)


def write_tex(path: str, rows: dict[str, Json]) -> None:
    labels = {
        "semantic_top1": "Semantic Top-1",
        "trace_rag": "Trace-RAG",
        "structmem_rag": "StructMem-RAG",
        "graphrag_static": "GraphRAG-static",
        "gems_no_reliability": "GEMS w/o Reliability",
        "gems": "\\textsc{GEMS}",
        "gems_reliability_only": "Reliability only",
    }
    best = max(row["reliable_at_1"] for row in rows.values()) if rows else 0.0
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{GEMS-targeted reliability stress benchmark derived from existing API candidates. The primary metric is selecting an observed-success endpoint under memory hazards.}",
        "\\label{tab:gems-stress-benchmark}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & Reliable@1 $\\uparrow$ & Failed@1 $\\downarrow$ & Success MRR $\\uparrow$ & Workflow Reliable $\\uparrow$ & Gold API Acc $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for method, row in rows.items():
        reliable = fmt(row["reliable_at_1"])
        if abs(row["reliable_at_1"] - best) < 1e-12:
            reliable = f"\\textbf{{{reliable}}}"
        lines.append(
            f"{labels.get(method, method)} & {reliable} & {fmt(row['failed_at_1'])} & "
            f"{fmt(row['success_mrr'])} & {fmt(row['workflow_reliable'])} & {fmt(row['gold_api_acc'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _, examples, splits, scorer, endpoints_by_url = build_scorer(args)
    selected = select_examples(examples, splits["test"], endpoints_by_url, args.mode, args.max_examples)
    methods = [
        "semantic_top1",
        "trace_rag",
        "structmem_rag",
        "graphrag_static",
        "gems_no_reliability",
        "gems",
        "gems_reliability_only",
    ]
    rows = evaluate_methods(methods, selected, scorer, endpoints_by_url)
    tag_counts = Counter(tag for _, tags in selected for tag in tags)
    payload = {
        "config": vars(args),
        "dataset": {
            "examples": len(selected),
            "workflows": len({example.record_id for example, _ in selected}),
            "tag_counts": dict(tag_counts),
            "selection_rule": (
                "test split steps where semantic or memory hazard is present and at least one candidate has observed_success=True"
            ),
            "primary_metric": "reliable_at_1",
        },
        "rows": rows,
        "notes": [
            "This is a targeted stress benchmark, not a replacement for the clean public composition benchmark.",
            "Reliable@1 measures observed-success endpoint selection; Gold API Acc is reported separately to expose the tradeoff with original gold matching.",
        ],
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_tex(args.output_tex, rows)
    print(json.dumps({"dataset": payload["dataset"], "rows": {k: {m: v for m, v in row.items() if m != "predictions"} for k, row in rows.items()}}, ensure_ascii=False, indent=2))
    print(f"saved {args.output_json}")
    print(f"saved {args.output_tex}")


if __name__ == "__main__":
    main()
