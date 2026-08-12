#!/usr/bin/env python
"""Summarize LLM reranker result files against local baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*")
    parser.add_argument("--glob", default="outputs/llm_public_composition_rerank*.json")
    parser.add_argument("--baseline", default="outputs/public_composition_experiments.json")
    parser.add_argument("--output", default="outputs/llm_rerank_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    semantic_test = baseline["results"]["test"]["semantic_top1"]
    rows = []
    input_files = [Path(name) for name in args.files] if args.files else sorted(Path(".").glob(args.glob))
    for path in input_files:
        obj = json.loads(path.read_text(encoding="utf-8"))
        metrics = obj.get("metrics") or {}
        config = obj.get("config") or {}
        if config.get("model") == "gpt-4.1-mini" and any(
            (pred.get("parsed") or {}).get("rationale") == "dry_run"
            for pred in obj.get("predictions", [])
        ):
            continue
        requested_steps = metrics.get("requested_steps") or metrics.get("steps") or 0
        completed_steps = metrics.get("steps") or 0
        coverage = metrics.get("coverage")
        if coverage is None:
            coverage = completed_steps / requested_steps if requested_steps else 0.0
        paper_ready = (
            completed_steps == requested_steps
            and coverage >= 0.99
            and not config.get("cache_only")
            and (completed_steps >= 100 or config.get("case_filter") == "all")
        )
        rows.append(
            {
                "file": str(path),
                "split": config.get("split"),
                "case_filter": config.get("case_filter"),
                "limit": config.get("limit"),
                "model": config.get("model"),
                "steps": metrics.get("steps"),
                "api_acc": metrics.get("api_acc"),
                "workflow_exact": metrics.get("workflow_exact"),
                "para_f1": metrics.get("para_f1"),
                "mrr_proxy": metrics.get("single_choice_mrr_proxy"),
                "tokens": (obj.get("usage") or {}).get("total_tokens"),
                "elapsed_seconds": obj.get("elapsed_seconds"),
                "paper_ready": bool(paper_ready),
                "coverage": coverage,
                "requested_steps": requested_steps,
            }
        )
    payload = {
        "semantic_top1_test_baseline": {
            "api_acc": semantic_test["api_acc"],
            "workflow_exact": semantic_test["workflow_exact"],
            "para_f1": semantic_test["para_f1"],
            "api_mrr": semantic_test["api_mrr"],
        },
        "llm_runs": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
