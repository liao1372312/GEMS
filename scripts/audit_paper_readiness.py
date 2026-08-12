#!/usr/bin/env python
"""Audit whether local experiment artifacts are ready for paper claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-summary", default="outputs/llm_rerank_summary.json")
    parser.add_argument("--llm-router", default="outputs/llm_router_eval.json")
    parser.add_argument("--llm-acceptance-router", default="outputs/llm_acceptance_router_eval.json")
    parser.add_argument("--gems-verifier", default="outputs/gems_plan_verifier_eval.json")
    parser.add_argument("--gems-api-verifier", default="outputs/gems_plan_verifier_robust_api_eval.json")
    parser.add_argument("--gems-adaptive", default="outputs/gems_adaptive_verifier_eval.json")
    parser.add_argument("--main-static-results", default="outputs/main_static_composition_results.json")
    parser.add_argument("--missing-val-plan", default="outputs/missing_val_llm_plan.json")
    parser.add_argument("--tables", default="outputs/experiment_tables.tex")
    parser.add_argument("--output", default="outputs/paper_readiness_audit.json")
    parser.add_argument("--markdown-output", default="outputs/paper_readiness_audit.md")
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        return {"missing_file": str(file)}
    return json.loads(file.read_text(encoding="utf-8"))


def artifact_status(name: str, path: str, paper_ready: bool, reason: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "path": path,
        "paper_ready": bool(paper_ready),
        "reason": reason,
        "details": details or {},
    }


def llm_summary_status(path: str, obj: dict[str, Any]) -> dict[str, Any]:
    if obj.get("missing_file"):
        return artifact_status("LLM rerank summary", path, False, "missing summary file")
    rows = obj.get("llm_runs") or []
    val_rows = [row for row in rows if row.get("split") == "val" and row.get("case_filter") == "all"]
    test_rows = [row for row in rows if row.get("split") == "test" and row.get("case_filter") == "all"]
    val_ready = any(row.get("paper_ready") and row.get("coverage", 0.0) >= 0.99 for row in val_rows)
    test_ready = any(row.get("paper_ready") and row.get("coverage", 0.0) >= 0.99 for row in test_rows)
    return artifact_status(
        "LLM rerank summary",
        path,
        val_ready and test_ready,
        "full validation and test LLM rerank files are ready" if val_ready and test_ready else "missing full validation or test LLM rerank coverage",
        {"val_ready": val_ready, "test_ready": test_ready, "runs": len(rows)},
    )


def generic_ready_status(name: str, path: str, obj: dict[str, Any]) -> dict[str, Any]:
    if obj.get("missing_file"):
        return artifact_status(name, path, False, "missing artifact")
    return artifact_status(
        name,
        path,
        bool(obj.get("paper_ready")),
        obj.get("selection_note") or obj.get("note") or ("paper_ready=true" if obj.get("paper_ready") else "paper_ready=false"),
        {
            "val_coverage": obj.get("val_coverage"),
            "coverage": obj.get("coverage"),
            "selection_split": obj.get("selection_split"),
        },
    )


def missing_val_status(path: str, obj: dict[str, Any]) -> dict[str, Any]:
    if obj.get("missing_file"):
        return artifact_status("Validation LLM coverage", path, False, "missing coverage plan")
    missing = int(obj.get("missing", 0))
    return artifact_status(
        "Validation LLM coverage",
        path,
        missing == 0,
        "validation LLM coverage complete" if missing == 0 else f"validation LLM missing {missing} steps",
        {
            "val_steps": obj.get("val_steps"),
            "available": obj.get("available"),
            "missing": missing,
            "coverage": obj.get("coverage"),
        },
    )


def table_status(path: str) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        return artifact_status("Exported tables", path, False, "missing exported table file")
    text = file.read_text(encoding="utf-8")
    diagnostic_count = text.lower().count("diagnostic")
    blocking_patterns = [
        "Semantic-to-LLM router result. Diagnostic",
        "LLM acceptance router result. Diagnostic",
        "GEMS adaptive operating points. Diagnostic",
        "Single-policy GEMS-Adaptive result under the balanced objective. Diagnostic",
    ]
    blockers = [pattern for pattern in blocking_patterns if pattern in text]
    return artifact_status(
        "Exported tables",
        path,
        not blockers,
        (
            "main router/verifier captions are paper-ready"
            if not blockers
            else f"{len(blockers)} stale diagnostic marker(s) remain in main router/verifier captions"
        ),
        {"diagnostic_count": diagnostic_count, "blocking_patterns": blockers},
    )


def main_static_status(path: str, obj: dict[str, Any]) -> dict[str, Any]:
    if obj.get("missing_file"):
        return artifact_status("Main static composition table", path, False, "missing main static result file")
    rows = obj.get("rows") or []
    required_ready = {"Trace-RAG", "StructMem-RAG", "GraphRAG-static", "Reflexion-Memory", "\\textsc{GEMS}"}
    ready = {row.get("method") for row in rows if row.get("paper_ready")}
    missing = sorted(required_ready - ready)
    pending_llm = [
        row.get("method")
        for row in rows
        if row.get("method") in {"Direct-LLM", "CoT-LLM", "ReAct", "RestGPT-style", "MA-NoMem"}
        and not row.get("paper_ready")
    ]
    return artifact_status(
        "Main static composition table",
        path,
        not missing,
        (
            "completed local/experience-memory main rows are ready"
            if not missing
            else f"missing completed main rows: {', '.join(missing)}"
        ),
        {
            "rows": len(rows),
            "ready_rows": sorted(ready),
            "pending_llm_agent_rows": pending_llm,
        },
    )


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Paper Readiness Audit",
        "",
        f"Overall paper-ready: `{str(payload['overall_paper_ready']).lower()}`",
        "",
        "| Artifact | Ready | Reason |",
        "|---|---:|---|",
    ]
    for status in payload["statuses"]:
        ready = "yes" if status["paper_ready"] else "no"
        lines.append(f"| {status['name']} | {ready} | {status['reason']} |")
    lines.extend(["", "## Required Next Actions", ""])
    for action in payload["required_next_actions"]:
        lines.append(f"- {action}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    statuses = [
        missing_val_status(args.missing_val_plan, load_json(args.missing_val_plan)),
        llm_summary_status(args.llm_summary, load_json(args.llm_summary)),
        generic_ready_status("Semantic/LLM router", args.llm_router, load_json(args.llm_router)),
        generic_ready_status(
            "LLM acceptance router",
            args.llm_acceptance_router,
            load_json(args.llm_acceptance_router),
        ),
        generic_ready_status("GEMS plan verifier", args.gems_verifier, load_json(args.gems_verifier)),
        generic_ready_status("GEMS API verifier", args.gems_api_verifier, load_json(args.gems_api_verifier)),
        generic_ready_status("GEMS adaptive verifier", args.gems_adaptive, load_json(args.gems_adaptive)),
        main_static_status(args.main_static_results, load_json(args.main_static_results)),
        table_status(args.tables),
    ]
    required_actions = []
    for status in statuses:
        if status["paper_ready"]:
            continue
        if status["name"] == "Validation LLM coverage":
            required_actions.append("Run `bash outputs/missing_val_llm_commands.sh` in a network-enabled environment.")
        elif status["name"] == "Exported tables":
            required_actions.append("Regenerate tables after paper-ready router/verifier outputs are available.")
        else:
            required_actions.append(f"Resolve `{status['path']}`: {status['reason']}.")
    payload = {
        "overall_paper_ready": all(status["paper_ready"] for status in statuses),
        "statuses": statuses,
        "required_next_actions": required_actions,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(Path(args.markdown_output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved {output}")
    print(f"saved {args.markdown_output}")


if __name__ == "__main__":
    main()
