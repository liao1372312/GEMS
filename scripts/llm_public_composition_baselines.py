#!/usr/bin/env python
"""Prompt-style LLM baselines for public composition API selection.

These baselines approximate Direct-LLM, CoT-style, ReAct-style, RestGPT-style,
and MA-NoMem selection under the same candidate-list setting. They do not
execute external APIs; they select APIs from the provided top-10 candidates.
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

from llm_public_composition_rerank import (
    candidate_payload,
    evaluate_predictions,
    load_client,
    parse_json_response,
    selected_index_from_result,
)
from run_public_composition_experiments import load_composition_examples, split_records


Json = dict[str, Any]


BASELINE_PROMPTS = {
    "direct_llm": """You are a service composition model. Select the best API candidate for the subtask.
Return only JSON: {"selected_rank": 1, "confidence": 0.0, "rationale": "short"}""",
    "cot_llm": """You are a service composition model. Internally compare task intent, domain, parameters, and endpoint description before selecting.
Return only JSON: {"selected_rank": 1, "confidence": 0.0, "rationale": "short"}""",
    "react": """You are a ReAct-style API selection agent. Think in terms of intent, candidate observation, and final action, but output only the final JSON.
Return only JSON: {"selected_rank": 1, "confidence": 0.0, "rationale": "short"}""",
    "restgpt": """You are a REST API planning model. Prefer candidates whose REST endpoint, method, required parameters, and description best match the subtask.
Return only JSON: {"selected_rank": 1, "confidence": 0.0, "rationale": "short"}""",
    "ma_nomem": """You simulate a memory-free multi-agent service composer. Planner, provider, and executor must agree using only the current request and candidates.
Return only JSON: {"selected_rank": 1, "confidence": 0.0, "rationale": "short"}""",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--output", default="outputs/llm_public_composition_baselines.json")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--baselines", default="direct_llm,cot_llm,react,restgpt,ma_nomem")
    parser.add_argument("--limit", type=int, default=10, help="Limit workflows; 0 means all.")
    parser.add_argument("--max-steps", type=int, default=0, help="Limit total evaluated steps after workflow filtering; 0 means all.")
    parser.add_argument("--case-filter", choices=["all", "non_top1", "top1"], default="all")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle selected steps before applying --max-steps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache-only", action="store_true", help="Use cached responses only and mark missing entries.")
    parser.add_argument("--continue-on-error", action="store_true", help="Record API errors and continue instead of aborting.")
    return parser.parse_args()


def stable_hash(value: object) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def build_payload(example: Any) -> Json:
    return {
        "user_request": example.user_query,
        "request_domain": example.record_domain,
        "subtask": {
            "step_id": example.step_id,
            "description": example.step_description,
            "required_inputs": example.required_inputs,
        },
        "candidates": [candidate_payload(candidate) for candidate in example.candidates],
    }


def cache_path(cache_dir: Path, model: str, baseline: str, example: Any) -> Path:
    key = stable_hash(
        {
            "model": model,
            "baseline": baseline,
            "record_id": example.record_id,
            "step_id": example.step_id,
            "payload": build_payload(example),
        }
    )
    return cache_dir / f"{key}.json"


def call_llm(
    client: OpenAI,
    config: Json,
    baseline: str,
    example: Any,
    cache_dir: Path,
    force: bool,
) -> Json:
    path = cache_path(cache_dir, config["model"], baseline, example)
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": BASELINE_PROMPTS[baseline]},
            {"role": "user", "content": json.dumps(build_payload(example), ensure_ascii=False)},
        ],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    item = {
        "record_id": example.record_id,
        "step_id": example.step_id,
        "raw": raw,
        "parsed": parse_json_response(raw),
        "usage": response.usage.model_dump() if response.usage else {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
    return item


def missing_cache_result(example: Any) -> Json:
    return {
        "record_id": example.record_id,
        "step_id": example.step_id,
        "parsed": {"selected_rank": 1, "confidence": 0.0, "rationale": "missing_cache"},
        "usage": {},
        "missing_cache": True,
    }


def error_result(example: Any, exc: Exception) -> Json:
    return {
        "record_id": example.record_id,
        "step_id": example.step_id,
        "parsed": {"selected_rank": 1, "confidence": 0.0, "rationale": "api_error"},
        "usage": {},
        "error": f"{type(exc).__name__}: {exc}",
    }


def select_examples(args: argparse.Namespace) -> list[Any]:
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    split_examples = [example for example in examples if example.record_id in splits[args.split]]
    if args.case_filter == "non_top1":
        split_examples = [example for example in split_examples if example.gold_index != 0]
    elif args.case_filter == "top1":
        split_examples = [example for example in split_examples if example.gold_index == 0]
    if args.limit > 0:
        record_ids = []
        seen = set()
        for example in split_examples:
            if example.record_id not in seen:
                seen.add(example.record_id)
                record_ids.append(example.record_id)
            if len(record_ids) >= args.limit:
                break
        keep = set(record_ids)
        split_examples = [example for example in split_examples if example.record_id in keep]
    if args.shuffle:
        import random

        rng = random.Random(args.seed)
        rng.shuffle(split_examples)
    if args.max_steps > 0:
        split_examples = split_examples[: args.max_steps]
    return split_examples


def main() -> None:
    args = parse_args()
    baselines = [item.strip() for item in args.baselines.split(",") if item.strip()]
    unknown = [item for item in baselines if item not in BASELINE_PROMPTS]
    if unknown:
        raise ValueError(f"Unknown baselines: {unknown}")

    examples = select_examples(args)
    if args.dry_run or args.cache_only:
        load_dotenv()
        config = {
            "model": os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1200")),
            "cache_dir": os.getenv("LLM_CACHE_DIR", "refine-logs/llm_cache"),
        }
        client = None
    else:
        client, config = load_client()
    cache_dir = Path(args.cache_dir or config["cache_dir"]) / "llm_public_composition_baselines"
    start = time.perf_counter()
    results: dict[str, Any] = {}
    for baseline in baselines:
        predictions = []
        for idx, example in enumerate(examples, start=1):
            if args.dry_run:
                item = {
                    "record_id": example.record_id,
                    "step_id": example.step_id,
                    "parsed": {"selected_rank": 1, "confidence": 0.0, "rationale": "dry_run"},
                    "usage": {},
                }
            elif args.cache_only:
                path = cache_path(cache_dir, config["model"], baseline, example)
                item = json.loads(path.read_text(encoding="utf-8")) if path.exists() else missing_cache_result(example)
            else:
                assert client is not None
                try:
                    item = call_llm(client, config, baseline, example, cache_dir, args.force)
                except (OpenAIError, OSError, TimeoutError) as exc:
                    item = error_result(example, exc)
                    if not args.continue_on_error:
                        raise
            predictions.append(item)
            parsed = item.get("parsed") or {}
            print(
                f"{baseline} [{idx}/{len(examples)}] {example.record_id} step={example.step_id} "
                f"gold={example.gold_index + 1} pred={parsed.get('selected_rank')}"
            )
        metrics = evaluate_predictions(examples, predictions)
        usage = {
            "prompt_tokens": sum((pred.get("usage") or {}).get("prompt_tokens", 0) for pred in predictions),
            "completion_tokens": sum((pred.get("usage") or {}).get("completion_tokens", 0) for pred in predictions),
            "total_tokens": sum((pred.get("usage") or {}).get("total_tokens", 0) for pred in predictions),
        }
        results[baseline] = {"metrics": metrics, "usage": usage, "predictions": predictions}
    payload = {
        "config": {
            "split": args.split,
            "limit_workflows": args.limit,
            "max_steps": args.max_steps,
            "baselines": baselines,
            "case_filter": args.case_filter,
            "shuffle": args.shuffle,
                "model": config["model"],
                "temperature": config["temperature"],
                "cache_dir": str(cache_dir),
                "cache_only": args.cache_only,
                "continue_on_error": args.continue_on_error,
                "note": "Prompt-style baselines select from the provided top-10 candidates and do not execute APIs.",
            },
        "elapsed_seconds": time.perf_counter() - start,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {"metrics": v["metrics"], "usage": v["usage"]} for k, v in results.items()}, ensure_ascii=False, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
