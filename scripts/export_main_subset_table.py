#!/usr/bin/env python
"""Export API-selection metrics for a fixed workflow subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_gems_plan_verifier import Policy, evaluate as evaluate_gems, load_llm_predictions
from evaluate_reflexion_memory_baseline import build_reflections, evaluate as evaluate_reflexion
from llm_public_composition_rerank import evaluate_predictions
from run_public_composition_experiments import (
    f1_score,
    load_composition_examples,
    rank_from_scores,
    required_param_names,
    split_records,
)
from tune_public_composition_reranker import build_context, build_feature_rows, evaluate_feature_rows, tuned_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-ids", default="outputs/test53_stratified_workflows.json")
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--processed-endpoints", default="dataset/processed/endpoints.jsonl")
    parser.add_argument("--memory", default="outputs/gems_memory_train_only.json")
    parser.add_argument("--llm-agent-baselines", default="outputs/llm_public_composition_baselines_test_all.json")
    parser.add_argument("--llm-rerank-files", nargs="*", default=["outputs/llm_public_composition_rerank_val_all.json", "outputs/llm_public_composition_rerank_test_all.json"])
    parser.add_argument("--gems-policy-results", default="outputs/gems_plan_verifier_robust_api_eval.json")
    parser.add_argument("--output-json", default="outputs/main_api_selection_test53.json")
    parser.add_argument("--output-tex", default="outputs/main_api_selection_test53.tex")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--trace-top-k", type=int, default=16)
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    file = Path(path)
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}


def load_workflow_ids(path: str) -> set[str]:
    obj = load_json(path)
    ids = obj.get("workflow_ids") if isinstance(obj, dict) else obj
    return {str(item) for item in ids or []}


def fmt(value: object) -> str:
    if value is None:
        return "--"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def load_gems_policy(path: str) -> Policy:
    obj = load_json(path)
    policy = obj.get("best_policy") or {}
    if not policy:
        return Policy(0.95, 0.01, 0.0, 0.05, 0.0, 0.0, 1, 2, False, tuple(range(2, 11)))
    return Policy(
        confidence_min=float(policy.get("confidence_min", 0.95)),
        margin_max=float(policy.get("margin_max", 0.01)),
        margin_min=float(policy.get("margin_min", 0.0)),
        sim_drop_max=float(policy.get("sim_drop_max", 0.05)),
        sim_drop_min=float(policy.get("sim_drop_min", 0.0)),
        param_overlap_min=float(policy.get("param_overlap_min", 0.0)),
        max_changed_steps=int(policy.get("max_changed_steps", 1)),
        max_plan_steps=int(policy.get("max_plan_steps", 2)),
        require_llm_change=bool(policy.get("require_llm_change", False)),
        allowed_ranks=tuple(int(item) for item in policy.get("allowed_ranks", list(range(2, 11)))),
    )


def para_f1_for_weights(ctx: dict[str, Any], examples: list[Any], weights: dict[str, float]) -> float:
    values = []
    for example in examples:
        ranked = rank_from_scores(tuned_scores(ctx, example, weights))
        pred_index = ranked[0] if ranked else 0
        values.append(
            f1_score(
                required_param_names(example.candidates[pred_index]),
                required_param_names(example.gold_candidate),
            )
        )
    return sum(values) / len(values) if values else 0.0


def row(method: str, metrics: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "method": method,
        "workflow_exact": metrics.get("workflow_exact"),
        "api_acc": metrics.get("api_acc"),
        "para_f1": metrics.get("para_f1"),
        "steps": metrics.get("steps"),
        "workflows": metrics.get("workflows"),
        "source": source,
    }


def main() -> None:
    args = parse_args()
    workflow_ids = load_workflow_ids(args.workflow_ids)
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    train_examples = [example for example in examples if example.record_id in splits["train"]]
    test_examples = [example for example in examples if example.record_id in workflow_ids]
    test_keys = {(example.record_id, example.step_id) for example in test_examples}
    workflows = len({example.record_id for example in test_examples})
    steps = len(test_examples)

    rows = []
    llm = load_json(args.llm_agent_baselines)
    for key, label in [
        ("direct_llm", "Direct-LLM"),
        ("cot_llm", "CoT-LLM"),
        ("react", "ReAct"),
        ("restgpt", "RestGPT-style"),
        ("ma_nomem", "MA-NoMem"),
    ]:
        preds_by_key = {
            (str(pred.get("record_id")), int(pred.get("step_id"))): pred
            for pred in (((llm.get("results") or {}).get(key) or {}).get("predictions") or [])
        }
        predictions = [preds_by_key.get((example.record_id, example.step_id), {"missing_cache": True}) for example in test_examples]
        rows.append(row(label, evaluate_predictions(test_examples, predictions), args.llm_agent_baselines))

    ctx = build_context(args)
    subset_ctx_examples = [example for example in ctx["test_examples"] if (example.record_id, example.step_id) in test_keys]
    feature_rows = build_feature_rows(ctx, subset_ctx_examples)
    tuned = {
        "Trace-RAG": {"rank": 0.15, "sim": 1.0, "trace": 0.35, "graph": 0.0, "rel": 0.05, "risk": 0.0, "success": 0.0},
        "StructMem-RAG": {"rank": 0.10, "sim": 1.0, "trace": 0.20, "graph": 0.20, "rel": 0.10, "risk": 0.0, "success": 0.0},
        "GraphRAG-static": {"rank": 0.10, "sim": 1.0, "trace": 0.20, "graph": 0.35, "rel": 0.0, "risk": 0.0, "success": 0.0},
    }
    for label, weights in tuned.items():
        metrics = evaluate_feature_rows(feature_rows, weights)
        metrics["para_f1"] = para_f1_for_weights(ctx, test_examples, weights)
        metrics["steps"] = steps
        metrics["workflows"] = workflows
        rows.append(row(label, metrics, "semantic-safe tuned feature reranker"))

    reflections = build_reflections(train_examples)
    rows.append(
        row(
            "AgentKB-style",
            evaluate_reflexion(
                test_examples,
                reflections,
                top_k_reflections=8,
                sim_weight=1.0,
                reflection_weight=0.05,
                support_weight=0.0,
            ),
            "semantic-safe AgentKB-style rerun on subset",
        )
    )

    predictions = load_llm_predictions(args.llm_rerank_files)
    gems_policy = load_gems_policy(args.gems_policy_results)
    rows.append(row("\\textsc{GEMS}", evaluate_gems(test_examples, predictions, gems_policy), f"GEMS validation-selected policy from {args.gems_policy_results}"))

    best = {
        field: max(item[field] for item in rows if isinstance(item.get(field), (int, float)))
        for field in ["workflow_exact", "api_acc", "para_f1"]
    }

    def cell(item: dict[str, Any], field: str) -> str:
        value = item.get(field)
        text = fmt(value)
        return f"\\textbf{{{text}}}" if isinstance(value, (int, float)) and value == best[field] else text

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{API-selection performance on the stratified {workflows}-workflow pilot subset. Workflow Exact is not Plan.Acc.}}",
        "\\label{tab:api-selection-pilot}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & Workflow Exact $\\uparrow$ & API.Acc $\\uparrow$ & Para.F1 $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for item in rows:
        lines.append(f"{item['method']} & {cell(item, 'workflow_exact')} & {cell(item, 'api_acc')} & {cell(item, 'para_f1')} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""])

    payload = {
        "workflow_subset": args.workflow_ids,
        "workflows": workflows,
        "steps": steps,
        "rows": rows,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_tex).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved {args.output_json}")
    print(f"saved {args.output_tex}")


if __name__ == "__main__":
    main()
