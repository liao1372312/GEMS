#!/usr/bin/env python
"""Export the corrected main table with true decomposition-level Plan.Acc."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METHODS = [
    ("direct_llm", "Direct-LLM"),
    ("cot_llm", "CoT-LLM"),
    ("react", "ReAct"),
    ("restgpt", "RestGPT-style"),
    ("ma_nomem", "MA-NoMem"),
    ("trace_rag", "Trace-RAG"),
    ("structmem_rag", "StructMem-RAG"),
    ("graphrag_static", "GraphRAG-static"),
    ("agentkb", "AgentKB-style"),
    ("gems", "\\textsc{GEMS}"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-results", default="outputs/plan_decomposition_eval_test_all.json")
    parser.add_argument("--api-results", default="outputs/main_static_composition_results_enhanced.json")
    parser.add_argument("--output-json", default="outputs/main_end_to_end_results.json")
    parser.add_argument("--output-tex", default="outputs/main_end_to_end_table.tex")
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    file = Path(path)
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}


def fmt(value: object) -> str:
    if value is None:
        return "--"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def api_rows_by_label(api_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = api_results.get("rows") or []
    return {str(row.get("method")): row for row in rows}


def main() -> None:
    args = parse_args()
    plan = load_json(args.plan_results)
    api = load_json(args.api_results)
    api_by_label = api_rows_by_label(api)
    rows: list[dict[str, Any]] = []
    for key, label in METHODS:
        plan_result = ((plan.get("results") or {}).get(key) or {})
        plan_metrics = plan_result.get("metrics") or {}
        api_row = api_by_label.get(label) or {}
        row = {
            "method": label,
            "plan_acc": plan_metrics.get("plan_acc"),
            "plan_sem_f1": plan_metrics.get("plan_sem_f1"),
            "step_count_acc": plan_metrics.get("step_count_acc"),
            "api_acc": api_row.get("api_acc"),
            "para_f1": api_row.get("para_f1"),
            "workflow_exact": api_row.get("workflow_exact", api_row.get("plan_acc")),
            "plan_workflows": plan_metrics.get("workflows"),
            "api_workflows": api_row.get("workflows"),
            "plan_source": args.plan_results,
            "api_source": api_row.get("source"),
            "paper_ready": bool(
                plan_metrics.get("coverage", 0.0) >= 0.99
                and isinstance(api_row.get("api_acc"), (int, float))
            ),
        }
        rows.append(row)

    numeric_fields = ["plan_acc", "plan_sem_f1", "api_acc", "para_f1", "workflow_exact"]
    best = {}
    for field in numeric_fields:
        values = [row[field] for row in rows if isinstance(row.get(field), (int, float))]
        if values:
            best[field] = max(values)

    def cell(row: dict[str, Any], field: str) -> str:
        value = row.get(field)
        text = fmt(value)
        if isinstance(value, (int, float)) and best.get(field) == value:
            return f"\\textbf{{{text}}}"
        return text

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Corrected end-to-end service composition results. Plan.Acc evaluates generated task decomposition against the gold TaskList; Workflow Exact is the former all-step API exact-match metric.}",
        "\\label{tab:main-end-to-end-results}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & Plan.Acc $\\uparrow$ & Plan.SemF1 $\\uparrow$ & API.Acc $\\uparrow$ & Para.F1 $\\uparrow$ & Workflow Exact $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['method']} & {cell(row, 'plan_acc')} & {cell(row, 'plan_sem_f1')} & "
            f"{cell(row, 'api_acc')} & {cell(row, 'para_f1')} & {cell(row, 'workflow_exact')} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""])

    payload = {
        "paper_ready": all(row["paper_ready"] for row in rows),
        "note": "Plan.Acc comes from generated decomposition evaluation; Workflow Exact is retained only as API-sequence exact match.",
        "rows": rows,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_tex).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"paper_ready": payload["paper_ready"], "rows": rows}, ensure_ascii=False, indent=2))
    print(f"saved {args.output_json}")
    print(f"saved {args.output_tex}")


if __name__ == "__main__":
    main()
