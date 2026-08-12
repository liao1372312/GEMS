#!/usr/bin/env python
"""Export LaTeX tables from local experiment JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-results", default="outputs/public_composition_experiments.json")
    parser.add_argument("--reranker-results", default="outputs/public_composition_learned_reranker.json")
    parser.add_argument("--llm-summary", default="outputs/llm_rerank_summary.json")
    parser.add_argument("--llm-router", default="outputs/llm_router_eval.json")
    parser.add_argument("--llm-acceptance-router", default="outputs/llm_acceptance_router_eval.json")
    parser.add_argument("--llm-analysis", default="outputs/llm_vs_semantic_analysis.json")
    parser.add_argument("--gems-verifier", default="outputs/gems_plan_verifier_eval.json")
    parser.add_argument("--gems-api-verifier", default="outputs/gems_plan_verifier_robust_api_eval.json")
    parser.add_argument("--gems-adaptive", default="outputs/gems_adaptive_verifier_eval.json")
    parser.add_argument("--test211-operating-points", default="outputs/test211_operating_points.json")
    parser.add_argument("--test211-cv-acceptance", default="outputs/test211_cv_acceptance.json")
    parser.add_argument("--emp-results", default="outputs/emp_composition_exec_eval.json")
    parser.add_argument("--hazard-online", default="outputs/public_hazard_online_eval.json")
    parser.add_argument("--output", default="outputs/experiment_tables.tex")
    return parser.parse_args()


def fmt(value: object) -> str:
    if value is None:
        return "--"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    comp = json.loads(Path(args.composition_results).read_text(encoding="utf-8"))
    rerank = json.loads(Path(args.reranker_results).read_text(encoding="utf-8"))
    llm_summary = {}
    if Path(args.llm_summary).exists():
        llm_summary = json.loads(Path(args.llm_summary).read_text(encoding="utf-8"))
    llm_router = {}
    if Path(args.llm_router).exists():
        llm_router = json.loads(Path(args.llm_router).read_text(encoding="utf-8"))
    llm_acceptance_router = {}
    if Path(args.llm_acceptance_router).exists():
        llm_acceptance_router = json.loads(Path(args.llm_acceptance_router).read_text(encoding="utf-8"))
    llm_analysis = {}
    if Path(args.llm_analysis).exists():
        llm_analysis = json.loads(Path(args.llm_analysis).read_text(encoding="utf-8"))
    gems_verifier = {}
    if Path(args.gems_verifier).exists():
        gems_verifier = json.loads(Path(args.gems_verifier).read_text(encoding="utf-8"))
    gems_api_verifier = {}
    if Path(args.gems_api_verifier).exists():
        gems_api_verifier = json.loads(Path(args.gems_api_verifier).read_text(encoding="utf-8"))
    gems_adaptive = {}
    if Path(args.gems_adaptive).exists():
        gems_adaptive = json.loads(Path(args.gems_adaptive).read_text(encoding="utf-8"))
    test211_operating_points = {}
    if Path(args.test211_operating_points).exists():
        test211_operating_points = json.loads(Path(args.test211_operating_points).read_text(encoding="utf-8"))
    test211_cv_acceptance = {}
    if Path(args.test211_cv_acceptance).exists():
        test211_cv_acceptance = json.loads(Path(args.test211_cv_acceptance).read_text(encoding="utf-8"))
    emp = {}
    if Path(args.emp_results).exists():
        emp = json.loads(Path(args.emp_results).read_text(encoding="utf-8"))
    hazard_online = {}
    if Path(args.hazard_online).exists():
        hazard_online = json.loads(Path(args.hazard_online).read_text(encoding="utf-8"))
    lines: list[str] = []
    lines.append("% Auto-generated from local experiment outputs.")
    lines.append("% Unsupported paper metrics such as Exec.SR and Retry require execution logs and are not filled here.")
    lines.append("")

    if emp.get("dataset"):
        dataset = emp["dataset"]
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Statistics of the industrial EMP composition benchmark.}")
        lines.append("\\label{tab:emp-statistics}")
        lines.append("\\begin{tabular}{lrrrr}")
        lines.append("\\toprule")
        lines.append("Split & Requests & Steps & Avg. Steps & Dep. Records \\\\")
        lines.append("\\midrule")
        for split in ["train", "val", "test"]:
            split_row = dataset["splits"][split]
            avg_steps = split_row["steps"] / split_row["records"] if split_row["records"] else 0.0
            dep = dataset["dependency_records"] if split == "test" else "--"
            lines.append(f"{split.title()} & {split_row['records']} & {split_row['steps']} & {avg_steps:.2f} & {dep} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

        methods = [
            ("semantic_top1", "Semantic top-1"),
            ("trace_rag", "Trace-RAG"),
            ("gems_no_quality", "\\textsc{GEMS} w/o quality"),
            ("gems_quality", "\\textsc{GEMS} + quality"),
        ]
        emp_test = emp["results"]["test"]
        lines.append("\\begin{table*}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Industrial EMP composition results. Sim.Exec.SR is simulated from EMP interface quality metadata because live execution logs are unavailable.}")
        lines.append("\\label{tab:emp-composition}")
        lines.append("\\begin{tabular}{lcccccc}")
        lines.append("\\toprule")
        lines.append("Method & API.Acc & API.Top-3 & API.MRR & Workflow Exact & Sim.Exec.SR & Para.F1 \\\\")
        lines.append("\\midrule")
        for key, label in methods:
            row = emp_test[key]
            lines.append(
                f"{label} & {fmt(row['api_acc'])} & {fmt(row['api_top3'])} & "
                f"{fmt(row['api_mrr'])} & {fmt(row['workflow_exact'])} & "
                f"{fmt(row['sim_exec_sr'])} & {fmt(row['para_f1'])} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table*}")
        lines.append("")

    dataset = comp["dataset"]
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Statistics of the local public composition benchmark.}")
    lines.append("\\label{tab:local-public-statistics}")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Split & Requests & Steps & Avg. Steps & Dependency Records \\\\")
    lines.append("\\midrule")
    for split in ["train", "val", "test"]:
        split_row = dataset["splits"][split]
        avg_steps = split_row["steps"] / split_row["records"] if split_row["records"] else 0.0
        dep = dataset["dependency_records"] if split == "test" else "--"
        lines.append(f"{split.title()} & {split_row['records']} & {split_row['steps']} & {avg_steps:.2f} & {dep} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    if llm_summary.get("llm_runs"):
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{LLM reranker diagnostic results. Non-top1 means the gold API is not the semantic rank-1 candidate.}")
        lines.append("\\label{tab:local-llm-reranker}")
        lines.append("\\begin{tabular}{lrrrrr}")
        lines.append("\\toprule")
        lines.append("Run & Steps & API.Acc & Workflow Exact & Para.F1 & Tokens \\\\")
        lines.append("\\midrule")
        for row in llm_summary["llm_runs"]:
            label = f"{row.get('case_filter')}@{row.get('limit')}"
            if not row.get("paper_ready"):
                label += " (sanity)"
            lines.append(
                f"{label} & {row.get('steps')} & {fmt(row.get('api_acc'))} & "
                f"{fmt(row.get('workflow_exact'))} & {fmt(row.get('para_f1'))} & {row.get('tokens')} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    if llm_router.get("test"):
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        router_caption = "Semantic-to-LLM router result."
        if not llm_router.get("paper_ready"):
            router_caption += " Diagnostic only because validation routed examples have incomplete LLM coverage."
        lines.append(f"\\caption{{{router_caption}}}")
        lines.append("\\label{tab:local-llm-router}")
        lines.append("\\begin{tabular}{lrrrrrr}")
        lines.append("\\toprule")
        lines.append("Method & API.Acc & Workflow Exact & Para.F1 & LLM Calls & Cache Hits & Missing \\\\")
        lines.append("\\midrule")
        sem = llm_router["semantic_top1_test"]
        test = llm_router["test"]
        lines.append(
            f"Semantic top-1 & {fmt(sem['api_acc'])} & {fmt(sem['workflow_exact'])} & "
            f"{fmt(sem['para_f1'])} & {sem['llm_calls']} & {sem['llm_cache_hits']} & {sem['missing_llm']} \\\\"
        )
        lines.append(
            f"Router & {fmt(test['api_acc'])} & {fmt(test['workflow_exact'])} & "
            f"{fmt(test['para_f1'])} & {test['llm_calls']} & {test['llm_cache_hits']} & {test['missing_llm']} \\\\"
        )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    if llm_acceptance_router.get("router_test"):
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        caption = "LLM acceptance router result."
        if not llm_acceptance_router.get("paper_ready"):
            coverage = llm_acceptance_router.get("coverage") or {}
            caption += (
                " Diagnostic only because validation LLM coverage is incomplete "
                f"({coverage.get('val_available', 0)}/{coverage.get('val_steps', 0)} steps)."
            )
        lines.append(f"\\caption{{{caption}}}")
        lines.append("\\label{tab:local-llm-acceptance-router}")
        lines.append("\\begin{tabular}{lrrrr}")
        lines.append("\\toprule")
        lines.append("Method & API.Acc & Workflow Exact & Para.F1 & LLM Accept Rate \\\\")
        lines.append("\\midrule")
        rows = [
            ("Semantic top-1", llm_acceptance_router["semantic_top1_test"]),
            ("All-step LLM", llm_acceptance_router["all_llm_test"]),
            ("Confidence router", llm_acceptance_router["router_test"]),
        ]
        formal = llm_acceptance_router.get("validation_trained_acceptance")
        if formal:
            rows.append(("Validation-trained acceptance", formal["test"]))
        for label, row in rows:
            lines.append(
                f"{label} & {fmt(row['api_acc'])} & {fmt(row['workflow_exact'])} & "
                f"{fmt(row['para_f1'])} & {fmt(row.get('llm_accept_rate', 0.0))} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    if llm_analysis.get("counts"):
        counts = llm_analysis["counts"]
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Help/hurt decomposition of full-test LLM reranking against semantic top-1.}")
        lines.append("\\label{tab:local-llm-help-hurt}")
        lines.append("\\begin{tabular}{lr}")
        lines.append("\\toprule")
        lines.append("Outcome & Steps \\\\")
        lines.append("\\midrule")
        lines.append(f"Both correct & {counts.get('both_correct', 0)} \\\\")
        lines.append(f"LLM helped & {counts.get('llm_helped', 0)} \\\\")
        lines.append(f"LLM hurt & {counts.get('llm_hurt', 0)} \\\\")
        lines.append(f"Both wrong & {counts.get('both_wrong', 0)} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    if gems_verifier.get("gems_plan_verifier_test"):
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        caption = "GEMS adaptive operating points."
        if not gems_verifier.get("paper_ready") or not gems_api_verifier.get("paper_ready"):
            caption += " Diagnostic only because verifier policies are selected on test pending full validation LLM predictions."
        lines.append(f"\\caption{{{caption}}}")
        lines.append("\\label{tab:local-gems-operating-points}")
        lines.append("\\begin{tabular}{lrrrr}")
        lines.append("\\toprule")
        lines.append("Method & API.Acc & Plan.Acc & Para.F1 & LLM Change Rate \\\\")
        lines.append("\\midrule")
        rows = [
            (gems_verifier["semantic_top1_test"], "Semantic top-1"),
            (gems_verifier["all_llm_test"], "LLM reranker"),
            (gems_verifier["gems_plan_verifier_test"], "\\textsc{GEMS}-Plan"),
        ]
        if gems_api_verifier.get("gems_plan_verifier_test"):
            rows.append((gems_api_verifier["gems_plan_verifier_test"], "\\textsc{GEMS}-API"))
        for row, label in rows:
            lines.append(
                f"{label} & {fmt(row['api_acc'])} & {fmt(row['workflow_exact'])} & "
                f"{fmt(row['para_f1'])} & {fmt(row['llm_change_rate'])} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    if gems_adaptive.get("gems_plan_verifier_test"):
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        caption = "Single-policy GEMS-Adaptive result under the balanced objective."
        if not gems_adaptive.get("paper_ready"):
            caption += " Diagnostic only pending full validation LLM predictions."
        lines.append(f"\\caption{{{caption}}}")
        lines.append("\\label{tab:local-gems-adaptive}")
        lines.append("\\begin{tabular}{lrrrr}")
        lines.append("\\toprule")
        lines.append("Method & API.Acc & Plan.Acc & Para.F1 & LLM Change Rate \\\\")
        lines.append("\\midrule")
        for row, label in [
            (gems_adaptive["semantic_top1_test"], "Semantic top-1"),
            (gems_adaptive["all_llm_test"], "LLM reranker"),
            (gems_adaptive["gems_plan_verifier_test"], "\\textsc{GEMS}-Adaptive"),
        ]:
            lines.append(
                f"{label} & {fmt(row['api_acc'])} & {fmt(row['workflow_exact'])} & "
                f"{fmt(row['para_f1'])} & {fmt(row['llm_change_rate'])} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    if test211_operating_points.get("rows"):
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Diagnostic operating points on the 211-workflow public test split. Policies are selected on test and should be treated as diagnostic.}")
        lines.append("\\label{tab:test211-operating-points}")
        lines.append("\\begin{tabular}{lrrrr}")
        lines.append("\\toprule")
        lines.append("Method & API.Acc & Plan.Acc & Para.F1 & LLM Change \\\\")
        lines.append("\\midrule")
        for row in test211_operating_points["rows"]:
            metrics = row["metrics"]
            label = row["name"].replace("GEMS", "\\textsc{GEMS}")
            lines.append(
                f"{label} & {fmt(metrics['api_acc'])} & {fmt(metrics['workflow_exact'])} & "
                f"{fmt(metrics['para_f1'])} & {fmt(metrics['llm_change_rate'])} \\\\"
            )
        if test211_cv_acceptance.get("best_cv_acceptance"):
            metrics = test211_cv_acceptance["best_cv_acceptance"]["metrics"]
            lines.append(
                f"\\textsc{{GEMS}}-CV-Accept & {fmt(metrics['api_acc'])} & {fmt(metrics['workflow_exact'])} & "
                f"{fmt(metrics['para_f1'])} & {fmt(metrics['llm_change_rate'])} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    methods = [
        ("semantic_top1", "Semantic top-1"),
        ("trace_rag", "Trace-RAG"),
        ("structmem_rag", "StructMem-RAG"),
        ("graphrag_static", "GraphRAG-static"),
        ("gems_no_reliability", "GEMS w/o reliability"),
        ("gems", "\\textsc{GEMS} heuristic"),
    ]
    test = comp["results"]["test"]
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Local public composition results. Execution-dependent metrics are omitted because live execution traces are unavailable.}")
    lines.append("\\label{tab:local-public-composition}")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("Method & API.Acc $\\uparrow$ & API.Top-3 $\\uparrow$ & API.MRR $\\uparrow$ & Workflow Exact $\\uparrow$ & Para.F1 $\\uparrow$ \\\\")
    lines.append("\\midrule")
    for key, label in methods:
        row = test[key]
        lines.append(
            f"{label} & {fmt(row['api_acc'])} & {fmt(row['api_top3'])} & {fmt(row['api_mrr'])} "
            f"& {fmt(row['workflow_exact'])} & {fmt(row['para_f1'])} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    lines.append("")

    if hazard_online.get("hazard_results"):
        hazard_rows = hazard_online["hazard_results"]
        hazard_methods = [
            ("semantic_top1", "Semantic"),
            ("trace_rag", "Trace-RAG"),
            ("graphrag_static", "GraphRAG"),
            ("gems", "\\textsc{GEMS}"),
            ("gems_reliability_only", "Reliability-only"),
        ]
        lines.append("\\begin{table*}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Proxy memory-hazard results on the public composition test split. Values are API.Acc on each hazard slice; $n$ is the number of affected steps.}")
        lines.append("\\label{tab:local-hazard-proxy}")
        lines.append("\\begin{tabular}{lrrrrrr}")
        lines.append("\\toprule")
        lines.append("Hazard Slice & $n$ & Semantic & Trace-RAG & GraphRAG & \\textsc{GEMS} & Reliability-only \\\\")
        lines.append("\\midrule")
        for hazard, rows in hazard_rows.items():
            n = next(iter(rows.values()))["count"] if rows else 0
            values = [fmt(rows[key]["api_acc"]) for key, _ in hazard_methods]
            lines.append(f"{hazard} & {n} & " + " & ".join(values) + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table*}")
        lines.append("")

        lines.append("\\begin{table*}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Safe top-ranked endpoint rate on proxy memory-hazard slices. A safe endpoint has observed successful feedback when feedback is available.}")
        lines.append("\\label{tab:local-hazard-safe}")
        lines.append("\\begin{tabular}{lrrrrr}")
        lines.append("\\toprule")
        lines.append("Hazard Slice & Semantic & Trace-RAG & GraphRAG & \\textsc{GEMS} & Reliability-only \\\\")
        lines.append("\\midrule")
        for hazard, rows in hazard_rows.items():
            values = [fmt(rows[key]["safe_top1_rate"]) for key, _ in hazard_methods]
            lines.append(f"{hazard} & " + " & ".join(values) + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table*}")
        lines.append("")

    if hazard_online.get("online_batches"):
        batches = hazard_online["online_batches"]
        online_methods = [
            ("semantic_top1", "Semantic"),
            ("trace_rag", "Trace-RAG"),
            ("graphrag_static", "GraphRAG"),
            ("gems", "\\textsc{GEMS}"),
        ]
        lines.append("\\begin{table*}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Sequential test-batch analysis on the public composition benchmark. Values are Workflow Exact; batches are deterministic proxy time slices.}")
        lines.append("\\label{tab:local-online-batches}")
        lines.append("\\begin{tabular}{lrrrrr}")
        lines.append("\\toprule")
        lines.append("Method & $T_1$ & $T_2$ & $T_3$ & $T_4$ & $T_5$ \\\\")
        lines.append("\\midrule")
        for key, label in online_methods:
            values = [fmt(batches[f"T{i}"][key]["workflow_exact"]) for i in range(1, 6)]
            lines.append(f"{label} & " + " & ".join(values) + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table*}")
        lines.append("")

    if hazard_online.get("error_analysis"):
        errors = hazard_online["error_analysis"]
        error_types = [
            "Wrong domain/API selection",
            "Parameter binding mismatch",
            "Wrong API selection",
            "Stale or failed endpoint reuse",
        ]
        error_methods = [
            ("semantic_top1", "Semantic"),
            ("trace_rag", "Trace-RAG"),
            ("graphrag_static", "GraphRAG"),
            ("gems", "\\textsc{GEMS}"),
        ]
        lines.append("\\begin{table*}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Automatic error analysis on failed public test steps. Counts are computed over each method's failed steps.}")
        lines.append("\\label{tab:local-error-analysis}")
        lines.append("\\begin{tabular}{lrrrr}")
        lines.append("\\toprule")
        lines.append("Error Type & Semantic & Trace-RAG & GraphRAG & \\textsc{GEMS} \\\\")
        lines.append("\\midrule")
        for error_type in error_types:
            values = [str(errors[key]["error_counts"].get(error_type, 0)) for key, _ in error_methods]
            lines.append(f"{error_type} & " + " & ".join(values) + " \\\\")
        totals = [str(errors[key]["failures"]) for key, _ in error_methods]
        lines.append("Total failures & " + " & ".join(totals) + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table*}")
        lines.append("")

    rerank_methods = [
        ("semantic_top1", "Semantic top-1"),
        ("hybrid_semantic_logreg", "Hybrid + logistic"),
        ("hybrid_semantic_random_forest", "Hybrid + random forest"),
        ("hybrid_semantic_hist_gradient_boosting", "Hybrid + HGB"),
    ]
    rerank_test = rerank["results"]
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Supervised reranker results on the local public benchmark.}")
    lines.append("\\label{tab:local-reranker}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Method & API.Acc & Workflow Exact & Para.F1 & MRR \\\\")
    lines.append("\\midrule")
    for key, label in rerank_methods:
        row = rerank_test[key]["test"]
        lines.append(
            f"{label} & {fmt(row['api_acc'])} & {fmt(row['workflow_exact'])} "
            f"& {fmt(row['para_f1'])} & {fmt(row['api_mrr'])} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
