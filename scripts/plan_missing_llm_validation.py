#!/usr/bin/env python
"""Summarize missing validation LLM rerank predictions and write rerun commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_public_composition_experiments import load_composition_examples, split_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--llm-files", nargs="*", default=["outputs/llm_public_composition_rerank_val_cache_only.json"])
    parser.add_argument("--output", default="outputs/missing_val_llm_plan.json")
    parser.add_argument("--commands-output", default="outputs/missing_val_llm_commands.sh")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--chunk-size", type=int, default=100)
    return parser.parse_args()


def key(record_id: str, step_id: int) -> str:
    return f"{record_id}::{step_id}"


def load_available(files: list[str]) -> set[str]:
    available: set[str] = set()
    for name in files:
        path = Path(name)
        if not path.exists():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        for pred in obj.get("predictions") or []:
            if pred.get("missing_cache") or pred.get("error"):
                continue
            if (pred.get("parsed") or {}).get("rationale") == "dry_run":
                continue
            available.add(key(str(pred.get("record_id")), int(pred.get("step_id"))))
    return available


def main() -> None:
    args = parse_args()
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    val_examples = [example for example in examples if example.record_id in splits["val"]]
    available = load_available(args.llm_files)
    missing = [
        {
            "index": index,
            "record_id": example.record_id,
            "step_id": example.step_id,
            "gold_rank": example.gold_index + 1,
        }
        for index, example in enumerate(val_examples)
        if key(example.record_id, example.step_id) not in available
    ]
    chunks = [
        {
            "chunk": chunk_index,
            "start_missing_index": start,
            "size": len(missing[start : start + args.chunk_size]),
        }
        for chunk_index, start in enumerate(range(0, len(missing), args.chunk_size), start=1)
    ]
    payload: dict[str, Any] = {
        "val_steps": len(val_examples),
        "available": len(val_examples) - len(missing),
        "missing": len(missing),
        "coverage": (len(val_examples) - len(missing)) / len(val_examples) if val_examples else 0.0,
        "missing_steps": missing,
        "chunks": chunks,
        "note": (
            "llm_public_composition_rerank.py currently selects the first N validation examples. "
            "Use --limit 0 with --continue-on-error in a network-enabled environment to fill all missing cache entries."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Run this in a network-enabled environment with LLM_API_KEY/OPENAI_API_KEY configured.",
        "python scripts/llm_public_composition_rerank.py \\",
        "  --split val \\",
        "  --limit 0 \\",
        "  --case-filter all \\",
        "  --continue-on-error \\",
        "  --output outputs/llm_public_composition_rerank_val_all.json",
        "",
        "python scripts/evaluate_llm_acceptance_router.py \\",
        "  --llm-files outputs/llm_public_composition_rerank_val_all.json outputs/llm_public_composition_rerank_test_all.json \\",
        "  --output outputs/llm_acceptance_router_eval.json",
        "",
        "python scripts/evaluate_gems_plan_verifier.py \\",
        "  --llm-files outputs/llm_public_composition_rerank_val_all.json outputs/llm_public_composition_rerank_test_all.json \\",
        "  --output outputs/gems_plan_verifier_eval.json",
        "",
        "python scripts/evaluate_gems_plan_verifier.py \\",
        "  --objective api \\",
        "  --llm-files outputs/llm_public_composition_rerank_val_all.json outputs/llm_public_composition_rerank_test_all.json \\",
        "  --output outputs/gems_plan_verifier_api_objective_eval.json",
        "",
        "python scripts/export_experiment_tables.py",
        "",
        "python scripts/audit_paper_readiness.py",
        "",
    ]
    command_path = Path(args.commands_output)
    command_path.parent.mkdir(parents=True, exist_ok=True)
    command_path.write_text("\n".join(commands), encoding="utf-8")
    command_path.chmod(0o755)
    print(json.dumps({k: payload[k] for k in ["val_steps", "available", "missing", "coverage"]}, ensure_ascii=False, indent=2))
    print(f"saved {output}")
    print(f"saved {command_path}")


if __name__ == "__main__":
    main()
