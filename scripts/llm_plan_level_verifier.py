#!/usr/bin/env python
"""Plan-level LLM verifier for public composition reranking.

This script asks an LLM to compare the semantic top-1 plan with the step-level
LLM reranked plan and decide, for the whole workflow, which steps should accept
the LLM change. It is designed to reduce plan-level damage from independent
step reranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_public_composition_rerank import load_client, selected_index_from_result
from run_public_composition_experiments import (
    f1_score,
    load_composition_examples,
    required_param_names,
    split_records,
)


Json = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--step-llm-result", default="outputs/llm_public_composition_rerank_test_all.json")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", default="outputs/llm_plan_level_verifier_test.json")
    parser.add_argument("--limit", type=int, default=0, help="Limit workflows, 0 means all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--risk-gate",
        choices=["none", "min-margin"],
        default="none",
        help="Only call the plan verifier for workflows that pass this risk gate.",
    )
    parser.add_argument("--margin-threshold", type=float, default=0.002)
    return parser.parse_args()


def key(record_id: str, step_id: int) -> str:
    return f"{record_id}::{step_id}"


def cache_key(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def candidate_brief(candidate: Json) -> Json:
    return {
        "api_name": candidate.get("api_name"),
        "domain": candidate.get("domain"),
        "description": candidate.get("description"),
        "endpoint": candidate.get("endpoint"),
        "method": candidate.get("method"),
        "required_parameters": [
            {
                "name": p.get("name"),
                "type": p.get("type"),
                "description": p.get("description"),
            }
            for p in candidate.get("required_parameters", [])
        ],
    }


def build_prompt(record_id: str, examples: list[Any], step_predictions: dict[str, Json]) -> list[dict[str, str]]:
    workflow = []
    for example in examples:
        pred = step_predictions.get(key(example.record_id, example.step_id))
        llm_index = selected_index_from_result(pred, len(example.candidates)) if pred else 0
        workflow.append(
            {
                "step_id": example.step_id,
                "step_description": example.step_description,
                "required_inputs": example.required_inputs,
                "semantic_top1_rank": 1,
                "semantic_top1": candidate_brief(example.candidates[0]),
                "llm_choice_rank": llm_index + 1,
                "llm_choice": candidate_brief(example.candidates[llm_index]),
                "llm_confidence": (pred.get("parsed") or {}).get("confidence") if pred else None,
                "llm_rationale": (pred.get("parsed") or {}).get("rationale") if pred else None,
            }
        )
    system = (
        "You are a strict workflow verifier for API composition. "
        "Your job is to preserve correct semantic top-1 choices and accept an LLM rerank only when it clearly better matches the step and workflow. "
        "Prefer minimal changes. A workflow fails if any step API is wrong."
    )
    user = {
        "record_id": record_id,
        "task": examples[0].user_query,
        "workflow_steps": workflow,
        "decision_rules": [
            "Return JSON only.",
            "For each step, set accept_llm=true only if the LLM choice is clearly more appropriate than semantic top-1.",
            "Reject LLM changes that only weakly improve wording, change to a broader unrelated API, or break the workflow intent.",
            "If unsure, keep semantic top-1.",
        ],
        "output_schema": {
            "decisions": [
                {
                    "step_id": "integer",
                    "accept_llm": "boolean",
                    "reason": "short string",
                }
            ],
            "workflow_confidence": "number from 0 to 1",
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def parse_json_response(text: str) -> Json:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    return json.loads(text)


def evaluate(examples_by_record: dict[str, list[Any]], step_predictions: dict[str, Json], plan_predictions: list[Json]) -> dict[str, Any]:
    plan_by_record = {item["record_id"]: item for item in plan_predictions}
    step_hits = 0
    workflow_hits = 0
    para_f1 = []
    llm_changed = 0
    total_steps = 0
    for record_id, examples in examples_by_record.items():
        plan = plan_by_record.get(record_id, {})
        decisions = {
            int(item.get("step_id")): bool(item.get("accept_llm"))
            for item in plan.get("parsed", {}).get("decisions", [])
            if item.get("step_id") is not None
        }
        workflow_ok = True
        for example in examples:
            pred = step_predictions.get(key(example.record_id, example.step_id))
            llm_index = selected_index_from_result(pred, len(example.candidates)) if pred else 0
            pred_index = llm_index if decisions.get(example.step_id, False) else 0
            llm_changed += int(pred_index != 0)
            correct = pred_index == example.gold_index
            step_hits += int(correct)
            workflow_ok = workflow_ok and correct
            para_f1.append(
                f1_score(
                    required_param_names(example.candidates[pred_index]),
                    required_param_names(example.gold_candidate),
                )
            )
            total_steps += 1
        workflow_hits += int(workflow_ok)
    workflows = len(examples_by_record)
    return {
        "api_acc": step_hits / total_steps if total_steps else 0.0,
        "workflow_exact": workflow_hits / workflows if workflows else 0.0,
        "para_f1": sum(para_f1) / len(para_f1) if para_f1 else 0.0,
        "llm_change_rate": llm_changed / total_steps if total_steps else 0.0,
        "steps": total_steps,
        "workflows": workflows,
    }


def workflow_min_margin(examples: list[Any]) -> float:
    margins = []
    for example in examples:
        if len(example.candidates) < 2:
            margins.append(1.0)
            continue
        first = float(example.candidates[0].get("similarity_score") or 0.0)
        second = float(example.candidates[1].get("similarity_score") or 0.0)
        margins.append(first - second)
    return min(margins) if margins else 1.0


def passes_risk_gate(examples: list[Any], gate: str, margin_threshold: float) -> bool:
    if gate == "none":
        return True
    if gate == "min-margin":
        return workflow_min_margin(examples) <= margin_threshold
    raise ValueError(f"Unknown risk gate: {gate}")


def semantic_keep_prediction(record_id: str, reason: str) -> Json:
    return {
        "record_id": record_id,
        "raw": "{}",
        "parsed": {"decisions": [], "workflow_confidence": 1.0, "gate_reason": reason},
        "usage": {},
        "skipped_by_gate": True,
    }


def call_plan_llm(messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
    client, config = load_client()
    response = client.chat.completions.create(
        model=config["model"],
        messages=messages,
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or "{}", response.usage.model_dump() if response.usage else {}


def main() -> None:
    args = parse_args()
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    split_examples = [example for example in examples if example.record_id in splits[args.split]]
    examples_by_record: dict[str, list[Any]] = {}
    for example in split_examples:
        examples_by_record.setdefault(example.record_id, []).append(example)
    examples_by_record = {
        record_id: sorted(items, key=lambda item: item.step_id)
        for record_id, items in examples_by_record.items()
    }
    if args.limit:
        examples_by_record = dict(list(examples_by_record.items())[: args.limit])

    step_obj = json.loads(Path(args.step_llm_result).read_text(encoding="utf-8"))
    step_predictions = {
        key(str(pred["record_id"]), int(pred["step_id"])): pred
        for pred in step_obj.get("predictions", [])
    }

    cache_dir = Path(args.cache_dir or os.getenv("LLM_CACHE_DIR", "refine-logs/llm_cache")) / "llm_plan_level_verifier"
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    start = time.time()
    for record_id, plan_examples in examples_by_record.items():
        if not passes_risk_gate(plan_examples, args.risk_gate, args.margin_threshold):
            outputs.append(semantic_keep_prediction(record_id, f"{args.risk_gate}>threshold"))
            continue
        messages = build_prompt(record_id, plan_examples, step_predictions)
        ckey = cache_key(messages)
        cache_path = cache_dir / f"{ckey}.json"
        if cache_path.exists():
            item = json.loads(cache_path.read_text(encoding="utf-8"))
        elif args.dry_run:
            item = {
                "record_id": record_id,
                "raw": "{}",
                "parsed": {"decisions": [], "workflow_confidence": 0.0},
                "usage": {},
            }
        else:
            raw, call_usage = call_plan_llm(messages)
            parsed = parse_json_response(raw)
            item = {"record_id": record_id, "raw": raw, "parsed": parsed, "usage": call_usage}
            cache_path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
        for field in usage:
            usage[field] += int((item.get("usage") or {}).get(field) or 0)
        outputs.append(item)
        if len(outputs) % 10 == 0:
            print(f"processed {len(outputs)}/{len(examples_by_record)} workflows")

    metrics = evaluate(examples_by_record, step_predictions, outputs)
    payload = {
        "config": vars(args),
        "metrics": metrics,
        "usage": usage,
        "elapsed_seconds": time.time() - start,
        "predictions": outputs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "usage": usage}, ensure_ascii=False, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
