#!/usr/bin/env python
"""LLM reranking for public composition candidates.

The script uses an OpenAI-compatible chat endpoint configured in `.env`.
It is intentionally cache-first and supports `--limit` for low-cost sanity
checks before full evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAIError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_public_composition_experiments import (
    f1_score,
    load_composition_examples,
    required_param_names,
    split_records,
)


SYSTEM_PROMPT = """You are an API selection judge for service composition.
Given a user request, one decomposed subtask, required inputs, and 10 candidate APIs,
select exactly one candidate API that best satisfies the subtask.

Use these criteria in order:
1. The API must directly satisfy the subtask intent.
2. Required parameters should be bindable from the user request, subtask, or required inputs.
3. Prefer specific APIs over broad generic search/content APIs when both are plausible.
4. Avoid APIs from unrelated domains even if they share words with the request.

Return only valid JSON with this schema:
{"selected_rank": 1, "confidence": 0.0, "rationale": "short reason"}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--output", default="outputs/llm_public_composition_rerank.json")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--case-filter",
        default="all",
        choices=["all", "non_top1", "top1_failed"],
        help="Restrict evaluation cases. non_top1 is useful for reranking diagnostics.",
    )
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Use cached LLM responses only. Missing cache entries are recorded and skipped.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record API errors and continue instead of aborting the run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write prompts and metadata without calling the LLM API.",
    )
    return parser.parse_args()


def stable_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_client() -> tuple[OpenAI, dict[str, Any]]:
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BASE_URL") or None
    if not api_key:
        raise RuntimeError("Missing LLM_API_KEY or OPENAI_API_KEY in environment")
    config = {
        "provider": os.getenv("LLM_PROVIDER", "openai"),
        "model": os.getenv("LLM_MODEL", "gpt-4.1-mini"),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0")),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1200")),
        "timeout": float(os.getenv("LLM_TIMEOUT_SECONDS", "180")),
        "cache_dir": os.getenv("LLM_CACHE_DIR", "refine-logs/llm_cache"),
    }
    return OpenAI(api_key=api_key, base_url=base_url, timeout=config["timeout"]), config


def candidate_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": candidate.get("rank"),
        "api_name": candidate.get("api_name"),
        "domain": candidate.get("domain"),
        "description": candidate.get("description"),
        "endpoint": candidate.get("endpoint"),
        "method": candidate.get("method"),
        "required_parameters": [
            {
                "name": param.get("name"),
                "type": param.get("type"),
                "description": param.get("description"),
                "default": param.get("default"),
            }
            for param in candidate.get("required_parameters") or []
        ],
    }


def build_prompt(example: Any) -> str:
    payload = {
        "user_request": example.user_query,
        "request_domain": example.record_domain,
        "subtask": {
            "step_id": example.step_id,
            "description": example.step_description,
            "required_inputs": example.required_inputs,
        },
        "candidates": [candidate_payload(candidate) for candidate in example.candidates],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def cache_path(cache_dir: Path, model: str, example: Any) -> Path:
    key = stable_hash(
        {
            "model": model,
            "record_id": example.record_id,
            "step_id": example.step_id,
            "prompt": build_prompt(example),
        }
    )
    return cache_dir / f"{key}.json"


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def call_llm(client: OpenAI, config: dict[str, Any], example: Any, cache_dir: Path, force: bool) -> dict[str, Any]:
    path = cache_path(cache_dir, config["model"], example)
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    prompt = build_prompt(example)
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    parsed = parse_json_response(content)
    result = {
        "record_id": example.record_id,
        "step_id": example.step_id,
        "raw": content,
        "parsed": parsed,
        "usage": response.usage.model_dump() if response.usage else {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def missing_cache_result(example: Any) -> dict[str, Any]:
    return {
        "record_id": example.record_id,
        "step_id": example.step_id,
        "parsed": {"selected_rank": 1, "confidence": 0.0, "rationale": "missing_cache"},
        "usage": {},
        "missing_cache": True,
    }


def error_result(example: Any, exc: Exception) -> dict[str, Any]:
    return {
        "record_id": example.record_id,
        "step_id": example.step_id,
        "parsed": {"selected_rank": 1, "confidence": 0.0, "rationale": "api_error"},
        "usage": {},
        "error": f"{type(exc).__name__}: {exc}",
    }


def selected_index_from_result(result: dict[str, Any], num_candidates: int) -> int:
    rank = (result.get("parsed") or {}).get("selected_rank", 1)
    try:
        index = int(rank) - 1
    except (TypeError, ValueError):
        index = 0
    if index < 0 or index >= num_candidates:
        return 0
    return index


def evaluate_predictions(examples: list[Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    hits = 0
    top3 = 0
    mrr = 0.0
    workflows: dict[str, list[bool]] = {}
    para_f1_values = []
    rank_counts: dict[int, int] = {}
    evaluated = 0
    missing_cache = 0
    api_errors = 0
    for example, pred in zip(examples, predictions):
        if pred.get("missing_cache"):
            missing_cache += 1
            continue
        if pred.get("error"):
            api_errors += 1
            continue
        evaluated += 1
        pred_index = selected_index_from_result(pred, len(example.candidates))
        rank_counts[pred_index + 1] = rank_counts.get(pred_index + 1, 0) + 1
        correct = pred_index == example.gold_index
        hits += int(correct)
        # LLM returns one choice. For top3/MRR, compare selected rank against gold
        # candidate list rank as a conservative single-choice diagnostic.
        top3 += int(example.gold_index < 3 if pred_index == 0 else correct)
        mrr += 1.0 if correct else 1.0 / (abs(pred_index - example.gold_index) + 2)
        workflows.setdefault(example.record_id, []).append(correct)
        para_f1_values.append(
            f1_score(
                required_param_names(example.candidates[pred_index]),
                required_param_names(example.gold_candidate),
            )
        )
    n = evaluated
    return {
        "api_acc": hits / n if n else 0.0,
        "single_choice_top3_proxy": top3 / n if n else 0.0,
        "single_choice_mrr_proxy": mrr / n if n else 0.0,
        "workflow_exact": sum(1 for values in workflows.values() if all(values)) / len(workflows) if workflows else 0.0,
        "para_f1": sum(para_f1_values) / len(para_f1_values) if para_f1_values else 0.0,
        "rank_counts": dict(sorted(rank_counts.items())),
        "steps": n,
        "requested_steps": len(examples),
        "missing_cache": missing_cache,
        "api_errors": api_errors,
        "coverage": n / len(examples) if examples else 0.0,
        "workflows": len(workflows),
    }


def save_payload(
    args: argparse.Namespace,
    config: dict[str, Any],
    cache_dir: Path,
    examples: list[Any],
    predictions: list[dict[str, Any]],
    elapsed: float,
    output: Path,
) -> None:
    metrics = evaluate_predictions(examples, predictions)
    usage = {
        "prompt_tokens": sum((pred.get("usage") or {}).get("prompt_tokens", 0) for pred in predictions),
        "completion_tokens": sum((pred.get("usage") or {}).get("completion_tokens", 0) for pred in predictions),
        "total_tokens": sum((pred.get("usage") or {}).get("total_tokens", 0) for pred in predictions),
    }
    payload = {
        "config": {
            "split": args.split,
            "limit": args.limit,
            "case_filter": args.case_filter,
            "model": config["model"],
            "temperature": config["temperature"],
            "cache_dir": str(cache_dir),
            "cache_only": args.cache_only,
            "continue_on_error": args.continue_on_error,
        },
        "metrics": metrics,
        "usage": usage,
        "elapsed_seconds": elapsed,
        "predictions": predictions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "usage": usage, "elapsed_seconds": elapsed}, ensure_ascii=False, indent=2))
    print(f"saved {output}")


def select_examples(args: argparse.Namespace) -> list[Any]:
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    selected = [example for example in examples if example.record_id in splits[args.split]]
    if args.case_filter == "non_top1":
        selected = [example for example in selected if example.gold_index != 0]
    elif args.case_filter == "top1_failed":
        # Endpoint feedback lookup is intentionally not used here to keep this
        # LLM script focused on semantic reranking. Use reliability scripts for
        # feedback-aware filtering.
        selected = [example for example in selected if example.gold_index != 0]
    if args.limit > 0:
        selected = selected[: args.limit]
    return selected


def main() -> None:
    args = parse_args()
    client, config = (None, None)
    if args.dry_run or args.cache_only:
        load_dotenv()
        config = {
            "model": os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0")),
            "cache_dir": os.getenv("LLM_CACHE_DIR", "refine-logs/llm_cache"),
        }
    else:
        client, config = load_client()
    cache_dir = Path(args.cache_dir or config["cache_dir"]) / "llm_public_composition_rerank"
    examples = select_examples(args)
    predictions: list[dict[str, Any]] = []
    start = time.perf_counter()
    for index, example in enumerate(examples, start=1):
        if args.dry_run:
            result = {
                "record_id": example.record_id,
                "step_id": example.step_id,
                "prompt": build_prompt(example),
                "parsed": {"selected_rank": 1, "confidence": 0.0, "rationale": "dry_run"},
                "usage": {},
            }
        elif args.cache_only:
            path = cache_path(cache_dir, config["model"], example)
            result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else missing_cache_result(example)
        else:
            try:
                result = call_llm(client, config, example, cache_dir, force=args.force)
            except (OpenAIError, OSError, TimeoutError) as exc:
                result = error_result(example, exc)
                if not args.continue_on_error:
                    predictions.append(result)
                    elapsed = time.perf_counter() - start
                    save_payload(args, config, cache_dir, examples, predictions, elapsed, Path(args.output))
                    raise
        predictions.append(result)
        parsed = result.get("parsed") or {}
        print(
            f"[{index}/{len(examples)}] {example.record_id} step={example.step_id} "
            f"gold={example.gold_index + 1} pred={parsed.get('selected_rank')} "
            f"conf={parsed.get('confidence')}"
        )
    elapsed = time.perf_counter() - start
    save_payload(args, config, cache_dir, examples, predictions, elapsed, Path(args.output))


if __name__ == "__main__":
    main()
