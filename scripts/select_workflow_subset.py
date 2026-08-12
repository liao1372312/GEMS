#!/usr/bin/env python
"""Select a deterministic stratified workflow subset for pilot experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_public_composition_experiments import load_composition_examples, split_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--sample-size", type=int, default=53)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--output", default="outputs/test53_stratified_workflows.json")
    return parser.parse_args()


def stable_score(seed: int, record_id: str) -> str:
    return hashlib.sha1(f"{seed}:{record_id}".encode("utf-8")).hexdigest()


def normalize_domain(domain: str) -> str:
    text = (domain or "unknown").strip().lower().replace("_", "-")
    aliases = {"healthcare": "medical", "tourism": "tourism", "technology": "technology", "finance": "finance"}
    return aliases.get(text, text)


def allocate_counts(strata: dict[tuple[str, str], list[str]], sample_size: int) -> dict[tuple[str, str], int]:
    total = sum(len(items) for items in strata.values())
    if sample_size >= total:
        return {key: len(items) for key, items in strata.items()}
    raw = {key: sample_size * len(items) / total for key, items in strata.items()}
    counts = {key: min(len(strata[key]), int(value)) for key, value in raw.items()}
    for key, items in strata.items():
        if counts[key] == 0 and items:
            counts[key] = 1
    while sum(counts.values()) > sample_size:
        candidates = [key for key, value in counts.items() if value > 1]
        key = max(candidates, key=lambda item: (counts[item] - raw[item], counts[item]))
        counts[key] -= 1
    while sum(counts.values()) < sample_size:
        candidates = [key for key, items in strata.items() if counts[key] < len(items)]
        key = max(candidates, key=lambda item: (raw[item] - counts[item], len(strata[item])))
        counts[key] += 1
    return counts


def main() -> None:
    args = parse_args()
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    record_by_id = {
        str(record.get("id")): record
        for record in records
        if str(record.get("id")) in splits[args.split]
    }
    examples_by_record: dict[str, list[Any]] = defaultdict(list)
    for example in examples:
        if example.record_id in record_by_id:
            examples_by_record[example.record_id].append(example)

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    workflow_rows = []
    for record_id, record in record_by_id.items():
        step_examples = examples_by_record.get(record_id, [])
        domain = normalize_domain(str(record.get("domain") or "unknown"))
        difficulty = "hard" if any(example.gold_index != 0 for example in step_examples) else "easy"
        strata[(domain, difficulty)].append(record_id)
        workflow_rows.append(
            {
                "record_id": record_id,
                "domain": domain,
                "difficulty": difficulty,
                "steps": len(step_examples),
                "non_top1_steps": sum(1 for example in step_examples if example.gold_index != 0),
            }
        )

    counts = allocate_counts(strata, args.sample_size)
    selected: list[str] = []
    for key, ids in strata.items():
        ordered = sorted(ids, key=lambda record_id: stable_score(args.seed, record_id))
        selected.extend(ordered[: counts[key]])
    selected = sorted(selected, key=lambda record_id: stable_score(args.seed, record_id))
    selected_set = set(selected)
    selected_rows = [row for row in workflow_rows if row["record_id"] in selected_set]

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "workflows": len(rows),
            "steps": sum(row["steps"] for row in rows),
            "hard_workflows": sum(1 for row in rows if row["difficulty"] == "hard"),
            "easy_workflows": sum(1 for row in rows if row["difficulty"] == "easy"),
            "non_top1_steps": sum(row["non_top1_steps"] for row in rows),
            "domain_counts": dict(sorted(Counter(row["domain"] for row in rows).items())),
            "strata_counts": {
                f"{domain}:{difficulty}": count
                for (domain, difficulty), count in sorted(Counter((row["domain"], row["difficulty"]) for row in rows).items())
            },
        }

    payload = {
        "config": vars(args),
        "selection": "deterministic stratified by normalized domain and top-1 difficulty",
        "workflow_ids": selected,
        "summary": summarize(selected_rows),
        "full_split_summary": summarize(workflow_rows),
        "rows": sorted(selected_rows, key=lambda row: selected.index(row["record_id"])),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
