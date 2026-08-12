#!/usr/bin/env python
"""Export a pilot table with semantic-safe strengthened baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_public_composition_experiments import f1_score, limit_workflows, rank_from_scores, required_param_names
from tune_public_composition_reranker import build_context, build_feature_rows, evaluate_feature_rows, tuned_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--processed-endpoints", default="dataset/processed/endpoints.jsonl")
    parser.add_argument("--memory", default="outputs/gems_memory_train_only.json")
    parser.add_argument("--reflexion", default="outputs/reflexion_memory_baseline_semantic_safe.json")
    parser.add_argument("--gems", default="outputs/gems_plan_verifier_safe_plan_eval.json")
    parser.add_argument("--llm-agent-baselines", default="outputs/llm_public_composition_baselines_test_all.json")
    parser.add_argument("--max-test-workflows", type=int, default=0)
    parser.add_argument("--workflow-ids", default="", help="Optional JSON file containing workflow_ids to evaluate.")
    parser.add_argument("--output-json", default="outputs/main_static_composition_results_enhanced.json")
    parser.add_argument("--output-tex", default="outputs/main_static_composition_table_enhanced.tex")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--trace-top-k", type=int, default=16)
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    file = Path(path)
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}


def load_workflow_ids(path: str) -> set[str]:
    if not path:
        return set()
    obj = load_json(path)
    ids = obj.get("workflow_ids") if isinstance(obj, dict) else obj
    return {str(item) for item in ids or []}


def metric_row(method: str, metrics: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "method": method,
        "workflow_exact": metrics.get("workflow_exact"),
        "api_acc": metrics.get("api_acc"),
        "para_f1": metrics.get("para_f1"),
        "steps": metrics.get("steps"),
        "workflows": metrics.get("workflows"),
        "source": source,
        "paper_ready": bool(metrics),
    }


def para_f1_for_weights(ctx: dict[str, Any], examples: list[Any], weights: dict[str, float]) -> float:
    values: list[float] = []
    for example in examples:
        scores = tuned_scores(ctx, example, weights)
        ranked = rank_from_scores(scores)
        pred_index = ranked[0] if ranked else 0
        values.append(
            f1_score(
                required_param_names(example.candidates[pred_index]),
                required_param_names(example.gold_candidate),
            )
        )
    return sum(values) / len(values) if values else 0.0


def public_metric(metrics: dict[str, float], steps: int, workflows: int, para_f1: float) -> dict[str, Any]:
    return {
        "workflow_exact": metrics["workflow_exact"],
        "api_acc": metrics["api_acc"],
        "para_f1": para_f1,
        "steps": steps,
        "workflows": workflows,
    }


def fmt(value: object) -> str:
    if value is None:
        return "--"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    ctx = build_context(args)
    workflow_ids = load_workflow_ids(args.workflow_ids)
    test_examples = [example for example in ctx["test_examples"] if example.record_id in workflow_ids] if workflow_ids else limit_workflows(ctx["test_examples"], args.max_test_workflows)
    feature_rows = build_feature_rows(ctx, test_examples)
    steps = len(test_examples)
    workflows = len({example.record_id for example in test_examples})

    tuned = {
        "Trace-RAG": {"rank": 0.15, "sim": 1.0, "trace": 0.35, "graph": 0.0, "rel": 0.05, "risk": 0.0, "success": 0.0},
        "StructMem-RAG": {"rank": 0.10, "sim": 1.0, "trace": 0.20, "graph": 0.20, "rel": 0.10, "risk": 0.0, "success": 0.0},
        "GraphRAG-static": {"rank": 0.10, "sim": 1.0, "trace": 0.20, "graph": 0.35, "rel": 0.0, "risk": 0.0, "success": 0.0},
    }

    rows: list[dict[str, Any]] = []
    llm = load_json(args.llm_agent_baselines)
    for key, label in [
        ("direct_llm", "Direct-LLM"),
        ("cot_llm", "CoT-LLM"),
        ("react", "ReAct"),
        ("restgpt", "RestGPT-style"),
        ("ma_nomem", "MA-NoMem"),
    ]:
        metrics = ((llm.get("results") or {}).get(key) or {}).get("metrics") or {}
        rows.append(metric_row(label, metrics, args.llm_agent_baselines))

    for label, weights in tuned.items():
        metrics = evaluate_feature_rows(feature_rows, weights)
        para_f1 = para_f1_for_weights(ctx, test_examples, weights)
        rows.append(metric_row(label, public_metric(metrics, steps, workflows, para_f1), "semantic-safe tuned feature reranker"))

    reflexion = load_json(args.reflexion)
    rows.append(metric_row("AgentKB-style", ((reflexion.get("results") or {}).get("test") or {}), args.reflexion))

    gems = load_json(args.gems)
    rows.append(metric_row("\\textsc{GEMS}", gems.get("gems_plan_verifier_test") or {}, args.gems))

    best = {
        field: max(row[field] for row in rows if isinstance(row.get(field), (int, float)))
        for field in ["workflow_exact", "api_acc", "para_f1"]
    }

    def cell(row: dict[str, Any], field: str) -> str:
        value = row[field]
        text = fmt(value)
        return f"\\textbf{{{text}}}" if isinstance(value, (int, float)) and value == best[field] else text

    caption = (
        f"Pilot performance on {workflows} test workflows with semantic-safe strengthened baselines."
        if args.max_test_workflows > 0
        else "Overall performance on the full public service composition test split with semantic-safe strengthened baselines."
    )
    note = (
        "Pilot table: non-GEMS memory baselines use semantic-safe validation-motivated weighting, not final paper evidence."
        if args.max_test_workflows > 0
        else "Full-test table: non-GEMS memory baselines use semantic-safe validation-motivated weighting."
    )

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:main-results-enhanced}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & Workflow Exact $\\uparrow$ & API.Acc $\\uparrow$ & Para.F1 $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(f"{row['method']} & {cell(row, 'workflow_exact')} & {cell(row, 'api_acc')} & {cell(row, 'para_f1')} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""])

    payload = {
        "split": "test",
        "workflows": workflows,
        "steps": steps,
        "note": note,
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
