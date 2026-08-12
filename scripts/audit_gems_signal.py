#!/usr/bin/env python
"""Audit whether GEMS graph-memory features carry useful composition signal."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_public_composition_experiments import (
    TraceRagScorer,
    add_composition_traces_to_memory,
    endpoint_key,
    ensure_candidate_nodes,
    load_composition_examples,
    load_processed_endpoint_maps,
    rank_from_scores,
    split_records,
    step_text,
)
from gems.graph_memory import ExecutionGraphMemory
from gems.retrieval import RoleSpecificRetriever


FEATURE_NAMES = [
    "semantic_similarity",
    "semantic_rank_reciprocal",
    "trace_score",
    "graph_score",
    "node_reliability",
    "node_risk",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--processed-endpoints", default="dataset/processed/endpoints.jsonl")
    parser.add_argument("--memory", default="outputs/gems_memory_train_only.json")
    parser.add_argument("--output", default="outputs/gems_signal_audit.json")
    parser.add_argument("--markdown-output", default="outputs/gems_signal_audit.md")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--trace-top-k", type=int, default=16)
    parser.add_argument(
        "--contrastive-memory",
        action="store_true",
        help="Add weak negative feedback to non-gold candidates from the training split.",
    )
    parser.add_argument("--positive-eta", type=float, default=0.03)
    parser.add_argument("--negative-eta", type=float, default=0.008)
    parser.add_argument("--negative-fail-credit", type=float, default=0.20)
    parser.add_argument("--max-negatives-per-step", type=int, default=9)
    return parser.parse_args()


def build_context(args: argparse.Namespace) -> dict[str, Any]:
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    split_examples = {
        split: [example for example in examples if example.record_id in ids]
        for split, ids in splits.items()
    }
    endpoints_by_url, _ = load_processed_endpoint_maps(args.processed_endpoints)
    memory = ExecutionGraphMemory.load(args.memory) if Path(args.memory).exists() else ExecutionGraphMemory()
    key_to_node_id = ensure_candidate_nodes(memory, examples, endpoints_by_url)
    add_composition_traces_to_memory(
        memory,
        split_examples["train"],
        key_to_node_id,
        contrastive_memory=args.contrastive_memory,
        positive_eta=args.positive_eta,
        negative_eta=args.negative_eta,
        negative_fail_credit=args.negative_fail_credit,
        max_negatives_per_step=args.max_negatives_per_step,
    )
    memory.propagate_reliability(layers=2)
    trace_scorer = TraceRagScorer(split_examples["train"])
    retriever = RoleSpecificRetriever(memory)
    return {
        "records": records,
        "examples": examples,
        "splits": splits,
        "split_examples": split_examples,
        "memory": memory,
        "key_to_node_id": key_to_node_id,
        "trace_scorer": trace_scorer,
        "retriever": retriever,
    }


def candidate_rows(ctx: dict[str, Any], examples: list[Any]) -> list[dict[str, Any]]:
    memory = ctx["memory"]
    key_to_node_id = ctx["key_to_node_id"]
    rows: list[dict[str, Any]] = []
    for example in examples:
        trace_scores, _ = ctx["trace_scorer"].candidate_scores(example, top_k=16)
        node_ids = [
            key_to_node_id.get(endpoint_key(candidate)) or key_to_node_id.get(str(candidate.get("api_name") or ""))
            for candidate in example.candidates
        ]
        valid_node_ids = [node_id for node_id in node_ids if node_id in memory.nodes]
        graph_scores = ctx["retriever"].score_nodes(step_text(example), "provider", valid_node_ids) if valid_node_ids else {}
        for idx, candidate in enumerate(example.candidates):
            node_id = node_ids[idx]
            node = memory.nodes.get(node_id or "")
            rows.append(
                {
                    "record_id": example.record_id,
                    "step_id": example.step_id,
                    "domain": example.record_domain,
                    "candidate_index": idx,
                    "label": int(idx == example.gold_index),
                    "semantic_similarity": float(candidate.get("similarity_score") or 0.0),
                    "semantic_rank_reciprocal": 1.0 / float(idx + 1),
                    "trace_score": max(
                        trace_scores.get(endpoint_key(candidate), 0.0),
                        trace_scores.get(str(candidate.get("api_name") or ""), 0.0),
                    ),
                    "graph_score": float(graph_scores.get(node_id or "", 0.0)),
                    "node_reliability": float(node.reliability) if node else 0.5,
                    "node_risk": float(node.risk) if node else 0.0,
                    "node_id": node_id,
                    "api_name": candidate.get("api_name"),
                    "endpoint_key": endpoint_key(candidate),
                }
            )
    return rows


def feature_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([row["label"] for row in rows], dtype=int)
    out: dict[str, Any] = {}
    for name in FEATURE_NAMES:
        scores = np.asarray([row[name] for row in rows], dtype=float)
        try:
            auc = float(roc_auc_score(y, scores))
        except ValueError:
            auc = None
        try:
            ap = float(average_precision_score(y, scores))
        except ValueError:
            ap = None
        positives = scores[y == 1]
        negatives = scores[y == 0]
        out[name] = {
            "roc_auc": auc,
            "average_precision": ap,
            "positive_mean": float(np.mean(positives)) if len(positives) else None,
            "negative_mean": float(np.mean(negatives)) if len(negatives) else None,
            "positive_median": float(np.median(positives)) if len(positives) else None,
            "negative_median": float(np.median(negatives)) if len(negatives) else None,
        }
    return out


def top1_by_feature(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["record_id"], row["step_id"])].append(row)
    out: dict[str, Any] = {}
    for name in FEATURE_NAMES:
        hits = 0
        top3 = 0
        mrr = 0.0
        workflows: dict[str, list[bool]] = defaultdict(list)
        for (record_id, _), group in groups.items():
            ranked = rank_from_scores([row[name] for row in group])
            labels = [row["label"] for row in group]
            gold_idx = labels.index(1)
            hit = ranked[0] == gold_idx
            hits += int(hit)
            top3 += int(gold_idx in ranked[:3])
            mrr += 1.0 / (ranked.index(gold_idx) + 1)
            workflows[record_id].append(hit)
        n = len(groups)
        out[name] = {
            "top1": hits / n if n else 0.0,
            "top3": top3 / n if n else 0.0,
            "mrr": mrr / n if n else 0.0,
            "workflow_exact": sum(1 for values in workflows.values() if all(values)) / len(workflows) if workflows else 0.0,
        }
    return out


def node_mapping_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    endpoint_to_nodes: dict[str, set[str]] = defaultdict(set)
    api_to_nodes: dict[str, set[str]] = defaultdict(set)
    node_to_candidates: dict[str, int] = Counter()
    missing = 0
    for row in rows:
        node_id = row.get("node_id")
        if not node_id:
            missing += 1
            continue
        endpoint_to_nodes[row["endpoint_key"]].add(node_id)
        api_to_nodes[str(row.get("api_name") or "")].add(node_id)
        node_to_candidates[node_id] += 1
    return {
        "candidate_rows": len(rows),
        "missing_node_rows": missing,
        "unique_endpoint_keys": len(endpoint_to_nodes),
        "unique_api_names": len(api_to_nodes),
        "unique_nodes": len(node_to_candidates),
        "api_name_collision_count": sum(1 for nodes in api_to_nodes.values() if len(nodes) > 1),
        "max_candidates_per_node": max(node_to_candidates.values()) if node_to_candidates else 0,
        "top_reused_nodes": node_to_candidates.most_common(10),
    }


def split_audit(ctx: dict[str, Any], split: str) -> dict[str, Any]:
    rows = candidate_rows(ctx, ctx["split_examples"][split])
    gold_rows = [row for row in rows if row["label"] == 1]
    non_gold_rows = [row for row in rows if row["label"] == 0]
    return {
        "steps": len(gold_rows),
        "candidate_rows": len(rows),
        "gold_rank_counts": dict(sorted(Counter(row["candidate_index"] + 1 for row in gold_rows).items())),
        "feature_metrics": feature_metrics(rows),
        "top1_by_feature": top1_by_feature(rows),
        "node_mapping": node_mapping_stats(rows),
        "gold_reliability_mean": float(np.mean([row["node_reliability"] for row in gold_rows])) if gold_rows else None,
        "non_gold_reliability_mean": float(np.mean([row["node_reliability"] for row in non_gold_rows])) if non_gold_rows else None,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# GEMS Signal Audit", ""]
    for split in ["val", "test"]:
        audit = payload["splits"][split]
        lines.extend(
            [
                f"## {split.title()}",
                "",
                f"- Steps: {audit['steps']}",
                f"- Candidate rows: {audit['candidate_rows']}",
                f"- Gold reliability mean: {fmt(audit['gold_reliability_mean'])}",
                f"- Non-gold reliability mean: {fmt(audit['non_gold_reliability_mean'])}",
                "",
                "| Feature | ROC-AUC | AP | Pos Mean | Neg Mean | Top-1 | Workflow Exact |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in FEATURE_NAMES:
            metric = audit["feature_metrics"][name]
            rank = audit["top1_by_feature"][name]
            lines.append(
                f"| {name} | {fmt(metric['roc_auc'])} | {fmt(metric['average_precision'])} | "
                f"{fmt(metric['positive_mean'])} | {fmt(metric['negative_mean'])} | "
                f"{fmt(rank['top1'])} | {fmt(rank['workflow_exact'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- If graph/reliability AUC is near 0.5 or lower than semantic similarity, the graph memory is not providing a useful gold-selection signal for this benchmark.",
            "- If reliability is similar for gold and non-gold candidates, reliability propagation cannot improve API selection without better execution labels.",
            "- Strong semantic-rank signal means the dataset is still dominated by the candidate generator's original ranking.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ctx = build_context(args)
    payload = {
        "config": vars(args),
        "splits": {
            "val": split_audit(ctx, "val"),
            "test": split_audit(ctx, "test"),
        },
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
