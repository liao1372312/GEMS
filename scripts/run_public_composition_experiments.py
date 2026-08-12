#!/usr/bin/env python
"""Run public composition experiments for the GEMS paper setup.

The script evaluates the parts of the paper experiment section that are
supported by the local datasets:

* RQ1: step-level API selection and workflow exact match.
* RQ2: memory-assisted API evidence retrieval.
* RQ3: failure-aware candidate selection as a memory-hazard proxy.
* RQ5: component ablations and lightweight efficiency metrics.

Metrics that require live execution logs, repair traces, or dense dataflow
annotations are reported as unsupported in the JSON output rather than
fabricated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gems.graph_memory import ExecutionGraphMemory, endpoint_api_node_id
from gems.retrieval import RoleSpecificRetriever
from gems.text import lexical_key, normalize_text


Json = dict[str, Any]


@dataclass
class StepExample:
    record_id: str
    record_domain: str
    user_query: str
    step_id: int
    step_description: str
    required_inputs: list[str]
    candidates: list[Json]
    gold_index: int
    gold_candidate: Json


@dataclass
class TraceItem:
    record_id: str
    step_id: int
    text: str
    endpoint_key: str
    api_name: str
    domain: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--processed-endpoints", default="dataset/processed/endpoints.jsonl")
    parser.add_argument("--memory", default="outputs/gems_memory_train_only.json")
    parser.add_argument("--output", default="outputs/public_composition_experiments.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--trace-top-k", type=int, default=16)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument(
        "--max-val-workflows",
        type=int,
        default=0,
        help="Limit validation workflows after the fixed train/val/test split; 0 means all.",
    )
    parser.add_argument(
        "--max-test-workflows",
        type=int,
        default=0,
        help="Limit test workflows after the fixed train/val/test split; 0 means all.",
    )
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


def stable_id(value: object, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def endpoint_key(candidate: Json) -> str:
    return str(candidate.get("endpoint") or candidate.get("api_name") or "")


def candidate_text(candidate: Json) -> str:
    params = candidate.get("required_parameters") or []
    param_text = " ".join(
        normalize_text(
            f"{param.get('name')} {param.get('type')} {param.get('description')} {param.get('default')}"
        )
        for param in params
    )
    return normalize_text(
        " ".join(
            str(part)
            for part in [
                candidate.get("api_name"),
                candidate.get("domain"),
                candidate.get("description"),
                candidate.get("method"),
                candidate.get("endpoint"),
                param_text,
            ]
            if part
        )
    )


def step_text(example: StepExample) -> str:
    return normalize_text(
        f"{example.user_query} {example.step_description} "
        f"{example.record_domain} {' '.join(example.required_inputs)}"
    )


def required_param_names(candidate: Json) -> set[str]:
    names: set[str] = set()
    for param in candidate.get("required_parameters") or []:
        if param.get("name"):
            names.add(str(param["name"]).strip().lower())
    return names


def f1_score(predicted: set[str], gold: set[str]) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    overlap = len(predicted & gold)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def selected_candidate(step: Json) -> Json | None:
    selection = step.get("selection") or {}
    candidate = selection.get("candidate")
    if isinstance(candidate, dict):
        return candidate
    rank = selection.get("selected_rank")
    try:
        rank_index = int(rank) - 1
    except (TypeError, ValueError):
        return None
    candidates = step.get("top_candidates") or []
    if 0 <= rank_index < len(candidates):
        return candidates[rank_index]
    return None


def selected_index(step: Json) -> int | None:
    gold = selected_candidate(step)
    if not gold:
        return None
    gold_key = endpoint_key(gold)
    gold_name = gold.get("api_name")
    for index, candidate in enumerate(step.get("top_candidates") or []):
        if endpoint_key(candidate) == gold_key or candidate.get("api_name") == gold_name:
            return index
    return None


def load_composition_examples(path: str | Path, max_records: int = 0) -> tuple[list[Json], list[StepExample]]:
    records: list[Json] = []
    examples: list[StepExample] = []
    for line in Path(path).open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        records.append(record)
        if max_records and len(records) >= max_records:
            break

    for record in records:
        record_id = str(record.get("id") or stable_id(record.get("task") or record.get("user_query")))
        user_query = str(record.get("user_query") or record.get("task") or "")
        record_domain = str(record.get("domain") or (record.get("metadata") or {}).get("paper_domain") or "unknown")
        task_list = record.get("TaskList") or []
        gold_steps = record.get("gold_api_candidates") or []
        for step_index, gold_step in enumerate(gold_steps):
            candidates = gold_step.get("top_candidates") or []
            gold_index = selected_index(gold_step)
            if gold_index is None or not candidates:
                continue
            task_step = task_list[step_index] if step_index < len(task_list) else {}
            examples.append(
                StepExample(
                    record_id=record_id,
                    record_domain=record_domain,
                    user_query=user_query,
                    step_id=int(gold_step.get("step_id") or task_step.get("id") or step_index + 1),
                    step_description=str(task_step.get("description") or ""),
                    required_inputs=[str(item) for item in task_step.get("required_inputs") or []],
                    candidates=candidates,
                    gold_index=gold_index,
                    gold_candidate=candidates[gold_index],
                )
            )
    return records, examples


def split_records(records: list[Json], seed: int, train_ratio: float, val_ratio: float) -> dict[str, set[str]]:
    ids = [str(record.get("id") or stable_id(record.get("task") or record.get("user_query"))) for record in records]
    ids = sorted(set(ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    train_end = int(len(ids) * train_ratio)
    val_end = train_end + int(len(ids) * val_ratio)
    return {
        "train": set(ids[:train_end]),
        "val": set(ids[train_end:val_end]),
        "test": set(ids[val_end:]),
    }


def load_processed_endpoint_maps(path: str | Path) -> tuple[dict[str, Json], dict[str, Json]]:
    by_url: dict[str, Json] = {}
    by_endpoint_id: dict[str, Json] = {}
    for line in Path(path).open(encoding="utf-8"):
        if not line.strip():
            continue
        endpoint = json.loads(line)
        if endpoint.get("url"):
            by_url[endpoint["url"]] = endpoint
        by_endpoint_id[endpoint["endpoint_id"]] = endpoint
    return by_url, by_endpoint_id


class TraceRagScorer:
    def __init__(self, train_examples: list[StepExample]) -> None:
        self.traces: list[TraceItem] = []
        for example in train_examples:
            gold = example.gold_candidate
            self.traces.append(
                TraceItem(
                    record_id=example.record_id,
                    step_id=example.step_id,
                    text=normalize_text(f"{step_text(example)} {candidate_text(gold)}"),
                    endpoint_key=endpoint_key(gold),
                    api_name=str(gold.get("api_name") or ""),
                    domain=str(gold.get("domain") or ""),
                )
            )
        docs = [trace.text or "empty" for trace in self.traces] or ["empty"]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=50000,
        )
        self.matrix = self.vectorizer.fit_transform(docs)

    def candidate_scores(self, example: StepExample, top_k: int) -> tuple[dict[str, float], list[TraceItem]]:
        if not self.traces:
            return {}, []
        query = self.vectorizer.transform([step_text(example) or "empty"])
        sims = cosine_similarity(query, self.matrix).ravel()
        top_indices = sorted(range(len(self.traces)), key=lambda idx: sims[idx], reverse=True)[:top_k]
        retrieved = [self.traces[idx] for idx in top_indices]
        scores: dict[str, float] = defaultdict(float)
        for idx in top_indices:
            trace = self.traces[idx]
            scores[trace.endpoint_key] = max(scores[trace.endpoint_key], float(sims[idx]))
            if trace.api_name:
                scores[trace.api_name] = max(scores[trace.api_name], float(sims[idx]) * 0.95)
        return dict(scores), retrieved


def ensure_candidate_nodes(
    memory: ExecutionGraphMemory,
    examples: list[StepExample],
    endpoints_by_url: dict[str, Json],
) -> dict[str, str]:
    key_to_node_id: dict[str, str] = {}
    for example in examples:
        for candidate in example.candidates:
            key = endpoint_key(candidate)
            endpoint = endpoints_by_url.get(candidate.get("endpoint"))
            if endpoint:
                node_id = endpoint_api_node_id(endpoint["endpoint_id"])
                if node_id not in memory.nodes:
                    memory.insert_endpoint(endpoint, {})
            else:
                synthetic_id = f"composition:{stable_id(key or candidate.get('api_name'))}"
                node_id = endpoint_api_node_id(synthetic_id)
                if node_id not in memory.nodes:
                    memory.add_node(
                        node_id,
                        "api",
                        candidate_text(candidate),
                        {
                            "endpoint_id": synthetic_id,
                            "service_name": str(candidate.get("api_name") or "").split(" - ", 1)[0],
                            "endpoint_name": str(candidate.get("api_name") or ""),
                            "category": candidate.get("domain"),
                            "method": candidate.get("method"),
                            "url": candidate.get("endpoint"),
                            "synthetic": True,
                        },
                        reliability=0.5,
                    )
            if key:
                key_to_node_id[key] = node_id
            if candidate.get("api_name"):
                key_to_node_id[str(candidate["api_name"])] = node_id
    return key_to_node_id


def add_composition_traces_to_memory(
    memory: ExecutionGraphMemory,
    train_examples: list[StepExample],
    key_to_node_id: dict[str, str],
    *,
    contrastive_memory: bool = False,
    positive_eta: float = 0.03,
    negative_eta: float = 0.008,
    negative_fail_credit: float = 0.20,
    max_negatives_per_step: int = 9,
) -> None:
    for example in train_examples:
        request_id = f"request:composition:{lexical_key(example.record_id, max_tokens=12)}"
        subtask_id = f"subtask:composition:{lexical_key(example.record_id, max_tokens=12)}:{example.step_id}"
        memory.add_node(
            request_id,
            "request",
            example.user_query,
            {"record_id": example.record_id, "domain": example.record_domain},
            reliability=0.75,
        )
        memory.add_node(
            subtask_id,
            "subtask",
            example.step_description or step_text(example),
            {
                "record_id": example.record_id,
                "step_id": example.step_id,
                "required_inputs": example.required_inputs,
                "domain": example.record_domain,
            },
            reliability=0.75,
        )
        try:
            memory.add_edge(request_id, subtask_id, "decomposes_to", reliability=0.75)
        except Exception:
            pass
        node_id = key_to_node_id.get(endpoint_key(example.gold_candidate)) or key_to_node_id.get(
            str(example.gold_candidate.get("api_name") or "")
        )
        if node_id and node_id in memory.nodes:
            memory.add_edge(subtask_id, node_id, "selects", {"composition_gold": True}, reliability=0.80)
            memory.apply_outcome_credit([request_id, subtask_id, node_id], success=True, eta=positive_eta)
            memory.nodes[node_id].attrs["composition_positive_count"] = (
                float(memory.nodes[node_id].attrs.get("composition_positive_count", 0.0)) + 1.0
            )

        if not contrastive_memory:
            continue

        negatives_added = 0
        for candidate_index, candidate in enumerate(example.candidates):
            if candidate_index == example.gold_index:
                continue
            negative_node_id = key_to_node_id.get(endpoint_key(candidate)) or key_to_node_id.get(
                str(candidate.get("api_name") or "")
            )
            if not negative_node_id or negative_node_id not in memory.nodes:
                continue
            _apply_weak_composition_negative(
                memory,
                negative_node_id,
                eta=negative_eta / max(1.0, float(candidate_index + 1) ** 0.5),
                fail_credit=negative_fail_credit,
            )
            negatives_added += 1
            if max_negatives_per_step > 0 and negatives_added >= max_negatives_per_step:
                break


def _apply_weak_composition_negative(
    memory: ExecutionGraphMemory,
    node_id: str,
    *,
    eta: float,
    fail_credit: float,
) -> None:
    node = memory.nodes.get(node_id)
    if not node:
        return
    # These are label negatives, not observed execution failures. Keep the risk
    # update much smaller than apply_outcome_credit(success=False).
    node.fail += max(0.0, fail_credit)
    node.risk = min(1.0, node.risk + max(0.0, eta))
    node.reliability = max(0.0, node.reliability - max(0.0, eta))
    node.attrs["composition_negative_count"] = float(node.attrs.get("composition_negative_count", 0.0)) + 1.0


class ExperimentScorer:
    def __init__(
        self,
        memory: ExecutionGraphMemory,
        trace_scorer: TraceRagScorer,
        key_to_node_id: dict[str, str],
        endpoints_by_url: dict[str, Json],
        trace_top_k: int,
    ) -> None:
        self.memory = memory
        self.trace_scorer = trace_scorer
        self.key_to_node_id = key_to_node_id
        self.endpoints_by_url = endpoints_by_url
        self.trace_top_k = trace_top_k
        self.retriever = RoleSpecificRetriever(memory)

    def score(self, method: str, example: StepExample) -> tuple[list[float], dict[str, Any]]:
        trace_scores, retrieved = self.trace_scorer.candidate_scores(example, top_k=self.trace_top_k)
        candidate_node_ids = [
            self.key_to_node_id.get(endpoint_key(candidate)) or self.key_to_node_id.get(str(candidate.get("api_name") or ""))
            for candidate in example.candidates
        ]
        valid_node_ids = [node_id for node_id in candidate_node_ids if node_id in self.memory.nodes]
        graph_scores = self.retriever.score_nodes(step_text(example), "provider", valid_node_ids) if valid_node_ids else {}

        scores: list[float] = []
        for index, candidate in enumerate(example.candidates):
            sim = float(candidate.get("similarity_score") or 0.0)
            trace = max(
                trace_scores.get(endpoint_key(candidate), 0.0),
                trace_scores.get(str(candidate.get("api_name") or ""), 0.0),
            )
            node_id = candidate_node_ids[index]
            node = self.memory.nodes.get(node_id or "")
            reliability = float(node.reliability) if node else 0.5
            risk = float(node.risk) if node else 0.0
            graph = float(graph_scores.get(node_id or "", 0.0))
            endpoint = self.endpoints_by_url.get(candidate.get("endpoint"))
            observed_success = endpoint.get("observed_success") if endpoint else None
            success_bonus = 0.0 if observed_success is None else (0.08 if observed_success is True else -0.08)

            if method == "semantic_top1":
                score = sim
            elif method == "trace_rag":
                score = 0.55 * trace + 0.45 * sim
            elif method == "structmem_rag":
                score = 0.65 * trace + 0.30 * sim + 0.05 * self._param_overlap_with_retrieved(candidate, retrieved)
            elif method == "graphrag_static":
                score = 0.50 * sim + 0.30 * graph + 0.20 * trace
            elif method == "gems_no_reliability":
                score = 0.50 * sim + 0.35 * trace + 0.15 * graph
            elif method == "gems_reliability_only":
                score = reliability - 0.25 * risk + success_bonus
            elif method == "gems":
                score = 0.40 * sim + 0.25 * trace + 0.20 * graph + 0.15 * reliability - 0.10 * risk
            elif method == "gems_contrastive":
                score = 0.42 * sim + 0.18 * trace + 0.12 * graph + 0.28 * reliability - 0.18 * risk
            elif method == "gems_oracle_gate":
                score = 0.45 * sim + 0.25 * trace + 0.15 * graph + 0.15 * reliability - 0.20 * risk + success_bonus
            else:
                raise ValueError(f"Unknown method: {method}")
            scores.append(float(score))
        return scores, {"retrieved": retrieved, "trace_scores": trace_scores}

    def _param_overlap_with_retrieved(self, candidate: Json, retrieved: list[TraceItem]) -> float:
        if not retrieved:
            return 0.0
        candidate_params = required_param_names(candidate)
        if not candidate_params:
            return 0.0
        # Structured-memory proxy: reward candidates whose API appeared in similar
        # traces and expose non-empty documented parameters.
        names = {trace.api_name for trace in retrieved[: self.trace_top_k]}
        return 1.0 if candidate.get("api_name") in names else min(0.5, len(candidate_params) / 4.0)


def rank_from_scores(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda idx: (scores[idx], -idx), reverse=True)


def evaluate_method(
    method: str,
    examples: list[StepExample],
    scorer: ExperimentScorer,
    endpoints_by_url: dict[str, Json],
) -> dict[str, Any]:
    start = time.perf_counter()
    step_hits = 0
    step_top3 = 0
    reciprocal_sum = 0.0
    param_f1_values: list[float] = []
    hallucinated = 0
    record_correct: dict[str, list[bool]] = defaultdict(list)
    record_domains: dict[str, str] = {}
    domain_hits: dict[str, list[int]] = defaultdict(list)
    harmful_predictions = 0
    known_feedback_predictions = 0
    hazard_subset_hits = 0
    hazard_subset_count = 0
    evidence_recall3 = 0
    evidence_count = 0

    for example in examples:
        scores, debug = scorer.score(method, example)
        ranked = rank_from_scores(scores)
        pred_index = ranked[0] if ranked else -1
        correct = pred_index == example.gold_index
        step_hits += int(correct)
        step_top3 += int(example.gold_index in ranked[:3])
        reciprocal_sum += 1.0 / (ranked.index(example.gold_index) + 1)
        record_correct[example.record_id].append(correct)
        record_domains[example.record_id] = example.record_domain
        domain_hits[example.record_domain].append(int(correct))

        pred_candidate = example.candidates[pred_index] if 0 <= pred_index < len(example.candidates) else {}
        if pred_candidate not in example.candidates:
            hallucinated += 1
        param_f1_values.append(f1_score(required_param_names(pred_candidate), required_param_names(example.gold_candidate)))

        endpoint = endpoints_by_url.get(pred_candidate.get("endpoint"))
        if endpoint and endpoint.get("observed_success") is not None:
            known_feedback_predictions += 1
            harmful_predictions += int(endpoint.get("observed_success") is False)

        candidate_feedback = [endpoints_by_url.get(candidate.get("endpoint")) for candidate in example.candidates]
        top_semantic_failed = bool(candidate_feedback and candidate_feedback[0] and candidate_feedback[0].get("observed_success") is False)
        has_success = any(endpoint and endpoint.get("observed_success") is True for endpoint in candidate_feedback)
        if top_semantic_failed and has_success:
            hazard_subset_count += 1
            hazard_subset_hits += int(endpoint is not None and endpoint.get("observed_success") is True)

        retrieved: list[TraceItem] = debug.get("retrieved") or []
        if retrieved:
            evidence_count += 1
            gold_key = endpoint_key(example.gold_candidate)
            gold_name = str(example.gold_candidate.get("api_name") or "")
            evidence_recall3 += int(
                any(trace.endpoint_key == gold_key or trace.api_name == gold_name for trace in retrieved[:3])
            )

    elapsed = time.perf_counter() - start
    steps = len(examples)
    workflow_hits = sum(1 for values in record_correct.values() if values and all(values))
    workflows = len(record_correct)
    return {
        "api_acc": step_hits / steps if steps else 0.0,
        "api_top3": step_top3 / steps if steps else 0.0,
        "api_mrr": reciprocal_sum / steps if steps else 0.0,
        "workflow_exact": workflow_hits / workflows if workflows else 0.0,
        "para_f1": statistics.mean(param_f1_values) if param_f1_values else 0.0,
        "hallu_rate": hallucinated / steps if steps else 0.0,
        "harmful_reuse_proxy": harmful_predictions / known_feedback_predictions if known_feedback_predictions else None,
        "hazard_success_proxy": hazard_subset_hits / hazard_subset_count if hazard_subset_count else None,
        "evidence_recall_at_3": evidence_recall3 / evidence_count if evidence_count else None,
        "avg_latency_ms_per_step": 1000.0 * elapsed / steps if steps else 0.0,
        "steps": steps,
        "workflows": workflows,
        "domain_api_acc": {
            domain: sum(values) / len(values)
            for domain, values in sorted(domain_hits.items())
            if values
        },
        "unsupported_metrics": {
            "exec_sr": "requires live or simulated workflow execution outcomes",
            "retry": "requires repair/retry traces",
            "recovery_rate": "requires temporal service-evolution repair logs",
        },
    }


def summarize_dataset(records: list[Json], examples: list[StepExample], splits: dict[str, set[str]]) -> dict[str, Any]:
    by_split = {
        split: [example for example in examples if example.record_id in ids]
        for split, ids in splits.items()
    }
    dependency_records = sum(1 for record in records if record.get("dependencies", record.get("dependency")))
    rank_counts = Counter(example.gold_index + 1 for example in examples)
    return {
        "records": len(records),
        "steps": len(examples),
        "avg_steps_per_record": len(examples) / len(records) if records else 0.0,
        "dependency_records": dependency_records,
        "dependency_record_rate": dependency_records / len(records) if records else 0.0,
        "selected_rank_counts": dict(sorted(rank_counts.items())),
        "splits": {
            split: {
                "records": len(ids),
                "steps": len(by_split[split]),
            }
            for split, ids in splits.items()
        },
    }


def limit_workflows(examples: list[StepExample], max_workflows: int) -> list[StepExample]:
    if max_workflows <= 0:
        return examples
    selected: list[StepExample] = []
    seen: set[str] = set()
    for example in examples:
        if example.record_id not in seen:
            if len(seen) >= max_workflows:
                break
            seen.add(example.record_id)
        selected.append(example)
    return selected


def main() -> None:
    args = parse_args()
    records, examples = load_composition_examples(args.composition_data, max_records=args.max_records)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    train_examples = [example for example in examples if example.record_id in splits["train"]]
    val_examples = [example for example in examples if example.record_id in splits["val"]]
    test_examples = [example for example in examples if example.record_id in splits["test"]]
    eval_val_examples = limit_workflows(val_examples, args.max_val_workflows)
    eval_test_examples = limit_workflows(test_examples, args.max_test_workflows)

    endpoints_by_url, _ = load_processed_endpoint_maps(args.processed_endpoints)
    if Path(args.memory).exists():
        memory = ExecutionGraphMemory.load(args.memory)
    else:
        memory = ExecutionGraphMemory()
    all_eval_examples = train_examples + val_examples + test_examples
    key_to_node_id = ensure_candidate_nodes(memory, all_eval_examples, endpoints_by_url)
    add_composition_traces_to_memory(
        memory,
        train_examples,
        key_to_node_id,
        contrastive_memory=args.contrastive_memory,
        positive_eta=args.positive_eta,
        negative_eta=args.negative_eta,
        negative_fail_credit=args.negative_fail_credit,
        max_negatives_per_step=args.max_negatives_per_step,
    )
    memory.propagate_reliability(layers=2)

    trace_scorer = TraceRagScorer(train_examples)
    scorer = ExperimentScorer(memory, trace_scorer, key_to_node_id, endpoints_by_url, args.trace_top_k)
    methods = [
        "semantic_top1",
        "trace_rag",
        "structmem_rag",
        "graphrag_static",
        "gems_no_reliability",
        "gems_reliability_only",
        "gems",
        "gems_contrastive",
        "gems_oracle_gate",
    ]

    results: dict[str, Any] = {}
    for split_name, split_examples in [("val", eval_val_examples), ("test", eval_test_examples)]:
        results[split_name] = {}
        for method in methods:
            results[split_name][method] = evaluate_method(method, split_examples, scorer, endpoints_by_url)

    payload = {
        "config": {
            "composition_data": args.composition_data,
            "processed_endpoints": args.processed_endpoints,
            "memory": args.memory,
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "trace_top_k": args.trace_top_k,
            "contrastive_memory": args.contrastive_memory,
            "positive_eta": args.positive_eta,
            "negative_eta": args.negative_eta,
            "negative_fail_credit": args.negative_fail_credit,
            "max_negatives_per_step": args.max_negatives_per_step,
            "max_val_workflows": args.max_val_workflows,
            "max_test_workflows": args.max_test_workflows,
        },
        "dataset": summarize_dataset(records, examples, splits),
        "results": results,
        "notes": [
            "Direct LLM, ReAct, RestGPT-style, and true Exec.SR require an LLM/execution backend and are not run by this local script.",
            "gems_oracle_gate uses endpoint observed_success as a diagnostic upper-bound gate, not a deployable transductive method.",
            "Hazard metrics are proxies based on endpoint observed_success because dense stale/schema-drift/repair labels are unavailable.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Dataset:", json.dumps(payload["dataset"], ensure_ascii=False, indent=2))
    for split_name in ["val", "test"]:
        print(f"\n{split_name.upper()} results")
        print(
            f"{'method':24s} {'api_acc':>8s} {'top3':>8s} {'mrr':>8s} "
            f"{'wf_exact':>9s} {'para_f1':>8s} {'haz_succ':>9s} {'lat_ms':>8s}"
        )
        for method in methods:
            row = results[split_name][method]
            hazard = row["hazard_success_proxy"]
            print(
                f"{method:24s} {row['api_acc']:8.4f} {row['api_top3']:8.4f} "
                f"{row['api_mrr']:8.4f} {row['workflow_exact']:9.4f} "
                f"{row['para_f1']:8.4f} "
                f"{(hazard if hazard is not None else 0.0):9.4f} "
                f"{row['avg_latency_ms_per_step']:8.3f}"
            )
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
