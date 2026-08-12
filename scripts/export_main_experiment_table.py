#!/usr/bin/env python
"""Export the main static-memory composition table from experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-results", default="outputs/public_composition_experiments.json")
    parser.add_argument("--gems-api-verifier", default="outputs/gems_plan_verifier_robust_api_eval.json")
    parser.add_argument("--reflexion", default="outputs/reflexion_memory_baseline.json")
    parser.add_argument("--llm-agent-baselines", default="outputs/llm_public_composition_baselines_test_all.json")
    parser.add_argument("--experience-label", default="AgentKB-style")
    parser.add_argument(
        "--allow-partial-llm",
        action="store_true",
        help="Report LLM-agent rows even when the LLM baseline file is a workflow-limited pilot run.",
    )
    parser.add_argument("--output-json", default="outputs/main_static_composition_results.json")
    parser.add_argument("--output-tex", default="outputs/main_static_composition_table.tex")
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding="utf-8"))


def fmt(value: object) -> str:
    if value is None:
        return "--"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def row_from_metrics(method: str, metrics: dict[str, Any] | None, *, source: str, paper_ready: bool) -> dict[str, Any]:
    metrics = metrics or {}
    return {
        "method": method,
        "workflow_exact": metrics.get("workflow_exact"),
        "api_acc": metrics.get("api_acc"),
        "para_f1": metrics.get("para_f1"),
        "exec_sr": metrics.get("exec_sr") or metrics.get("sim_exec_sr"),
        "hallu_rate": metrics.get("hallu_rate"),
        "retry": metrics.get("retry"),
        "steps": metrics.get("steps"),
        "workflows": metrics.get("workflows"),
        "source": source,
        "paper_ready": paper_ready,
    }


def main() -> None:
    args = parse_args()
    comp = load_json(args.composition_results)
    gems = load_json(args.gems_api_verifier)
    reflexion = load_json(args.reflexion)
    llm_agents = load_json(args.llm_agent_baselines)

    test = ((comp.get("results") or {}).get("test") or {})
    rows: list[dict[str, Any]] = []

    baseline_key_to_label = {
        "direct_llm": "Direct-LLM",
        "cot_llm": "CoT-LLM",
        "react": "ReAct",
        "restgpt": "RestGPT-style",
        "ma_nomem": "MA-NoMem",
    }
    llm_results = llm_agents.get("results") or {}
    config = llm_agents.get("config") or {}
    requested_full = config.get("split") == "test" and int(config.get("limit_workflows") or 0) == 0 and int(config.get("max_steps") or 0) == 0
    for key, label in baseline_key_to_label.items():
        obj = llm_results.get(key) or {}
        metrics = obj.get("metrics")
        coverage = float((metrics or {}).get("coverage") or 0.0)
        paper_ready = bool((requested_full or args.allow_partial_llm) and coverage >= 0.99)
        rows.append(row_from_metrics(label, metrics if paper_ready else None, source=args.llm_agent_baselines, paper_ready=paper_ready))

    rows.extend(
        [
            row_from_metrics("Trace-RAG", test.get("trace_rag"), source=args.composition_results, paper_ready=bool(test.get("trace_rag"))),
            row_from_metrics("StructMem-RAG", test.get("structmem_rag"), source=args.composition_results, paper_ready=bool(test.get("structmem_rag"))),
            row_from_metrics("GraphRAG-static", test.get("graphrag_static"), source=args.composition_results, paper_ready=bool(test.get("graphrag_static"))),
        ]
    )

    reflexion_test = ((reflexion.get("results") or {}).get("test") or {})
    rows.append(
        row_from_metrics(
            args.experience_label,
            reflexion_test if reflexion.get("paper_ready") else None,
            source=args.reflexion,
            paper_ready=bool(reflexion.get("paper_ready") and reflexion_test),
        )
    )

    gems_test = gems.get("gems_plan_verifier_test")
    rows.append(
        row_from_metrics(
            "\\textsc{GEMS}",
            gems_test,
            source=args.gems_api_verifier,
            paper_ready=bool(gems.get("paper_ready") and gems_test),
        )
    )

    numeric_fields = ["workflow_exact", "api_acc", "para_f1", "exec_sr", "hallu_rate", "retry"]
    best: dict[str, float] = {}
    for field in numeric_fields:
        ready_rows = [row for row in rows if row["paper_ready"]]
        values = [row[field] for row in ready_rows if isinstance(row[field], (int, float))]
        if values:
            best[field] = min(values) if field in {"hallu_rate", "retry"} else max(values)

    missing_llm_rows = [
        row["method"]
        for row in rows
        if row["method"] in {"Direct-LLM", "CoT-LLM", "ReAct", "RestGPT-style", "MA-NoMem"} and not row["paper_ready"]
    ]
    note = (
        "All rows use the full public test split."
        if not missing_llm_rows
        else "LLM-agent rows remain paper_ready=false until full-test prompt baseline output is provided."
    )

    payload = {
        "paper_ready": all(row["paper_ready"] for row in rows if row["method"] in {"Trace-RAG", "StructMem-RAG", "GraphRAG-static", args.experience_label, "\\textsc{GEMS}"})
        and all(row["paper_ready"] for row in rows if row["method"] not in {"Direct-LLM", "CoT-LLM", "ReAct", "RestGPT-style", "MA-NoMem"}),
        "note": note,
        "rows": rows,
    }

    def cell(row: dict[str, Any], field: str) -> str:
        value = row[field]
        text = fmt(value)
        if isinstance(value, (int, float)) and best.get(field) == value:
            return f"\\textbf{{{text}}}"
        return text

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Overall performance on the main service composition benchmark. Exec.SR and Retry require live execution or repair traces and are not available for the public benchmark.}",
        "\\label{tab:main-results}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Method & Workflow Exact $\\uparrow$ & API.Acc $\\uparrow$ & Para.F1 $\\uparrow$ & Exec.SR $\\uparrow$ & HalluRate $\\downarrow$ & Retry $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['method']} & {cell(row, 'workflow_exact')} & {cell(row, 'api_acc')} & {cell(row, 'para_f1')} & "
            f"{cell(row, 'exec_sr')} & {cell(row, 'hallu_rate')} & {cell(row, 'retry')} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""])

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_tex).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"paper_ready": payload["paper_ready"], "rows": rows}, ensure_ascii=False, indent=2))
    print(f"saved {args.output_json}")
    print(f"saved {args.output_tex}")


if __name__ == "__main__":
    main()
