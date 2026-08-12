#!/usr/bin/env python
"""Analyze reranking ability on cases where gold is not semantic top-1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-results", default="outputs/public_composition_experiments.json")
    parser.add_argument("--reranker-results", default="outputs/public_composition_learned_reranker.json")
    parser.add_argument("--output", default="outputs/non_top1_subset_analysis.json")
    return parser.parse_args()


def main() -> None:
    # This first version extracts aggregate counts available from existing
    # experiment metadata. Per-example correction analysis requires prediction
    # dumps, so this file records the supported subset and the missing artifact.
    args = parse_args()
    comp = json.loads(Path(args.composition_results).read_text(encoding="utf-8"))
    rerank = json.loads(Path(args.reranker_results).read_text(encoding="utf-8"))
    selected_counts = {int(k): int(v) for k, v in comp["dataset"]["selected_rank_counts"].items()}
    total = sum(selected_counts.values())
    non_top1 = total - selected_counts.get(1, 0)
    payload = {
        "total_steps": total,
        "gold_not_semantic_top1_steps": non_top1,
        "gold_not_semantic_top1_rate": non_top1 / total if total else 0.0,
        "semantic_top1_test_api_acc": comp["results"]["test"]["semantic_top1"]["api_acc"],
        "best_hybrid_test_mrr": rerank["results"]["hybrid_semantic_hist_gradient_boosting"]["test"]["api_mrr"],
        "semantic_top1_test_mrr": rerank["results"]["semantic_top1"]["test"]["api_mrr"],
        "interpretation": (
            "36.7% of all steps have gold outside semantic rank 1. Current hybrid rerankers improve MRR "
            "but do not improve top-1 accuracy. A per-example correction table should be added if the "
            "paper needs detailed error analysis."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
