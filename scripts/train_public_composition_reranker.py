#!/usr/bin/env python
"""Train supervised rerankers for the public composition benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_public_composition_experiments import (
    ExperimentScorer,
    TraceRagScorer,
    add_composition_traces_to_memory,
    candidate_text,
    endpoint_key,
    ensure_candidate_nodes,
    f1_score,
    load_composition_examples,
    load_processed_endpoint_maps,
    rank_from_scores,
    required_param_names,
    split_records,
    step_text,
)
from gems.graph_memory import ExecutionGraphMemory
from gems.retrieval import RoleSpecificRetriever


FEATURE_NAMES = [
    "candidate_rank_reciprocal",
    "candidate_rank_index",
    "candidate_similarity",
    "similarity_gap_to_top",
    "similarity_centered",
    "local_bm25",
    "local_bm25_gap_to_top",
    "local_word_tfidf",
    "local_word_tfidf_gap_to_top",
    "local_char_tfidf",
    "local_char_tfidf_gap_to_top",
    "trace_match",
    "trace_gap_to_top",
    "graph_score",
    "graph_gap_to_top",
    "node_reliability",
    "node_risk",
    "node_conflict",
    "param_name_overlap_with_required_inputs",
    "param_count",
    "has_required_param",
    "same_domain_as_task",
    "api_name_query_overlap",
    "candidate_text_query_overlap",
    "default_value_query_overlap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--processed-endpoints", default="dataset/processed/endpoints.jsonl")
    parser.add_argument("--memory", default="outputs/gems_memory_train_only.json")
    parser.add_argument("--output", default="outputs/public_composition_learned_reranker.json")
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


def token_set(text: object) -> set[str]:
    return {token for token in str(text or "").lower().replace("/", " ").replace("-", " ").split() if token}


def overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def local_text_scores(query: str, candidates: list[dict[str, Any]]) -> dict[str, list[float]]:
    docs = [candidate_text(candidate) or "empty" for candidate in candidates]
    tokenized_docs = [doc.lower().split() for doc in docs]
    query_tokens = query.lower().split()
    if tokenized_docs and query_tokens:
        bm25 = minmax([float(score) for score in BM25Okapi(tokenized_docs).get_scores(query_tokens)])
    else:
        bm25 = [0.0 for _ in docs]

    if docs:
        word_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1)
        word_matrix = word_vectorizer.fit_transform(docs + [query or "empty"])
        word_scores = cosine_similarity(word_matrix[-1], word_matrix[:-1]).ravel().astype(float).tolist()

        char_vectorizer = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        char_matrix = char_vectorizer.fit_transform(docs + [query or "empty"])
        char_scores = cosine_similarity(char_matrix[-1], char_matrix[:-1]).ravel().astype(float).tolist()
    else:
        word_scores = []
        char_scores = []

    return {
        "local_bm25": bm25,
        "local_word_tfidf": word_scores,
        "local_char_tfidf": char_scores,
    }


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
    scorer = ExperimentScorer(memory, trace_scorer, key_to_node_id, endpoints_by_url, args.trace_top_k)
    retriever = RoleSpecificRetriever(memory)
    return {
        "records": records,
        "examples": examples,
        "splits": splits,
        "split_examples": split_examples,
        "endpoints_by_url": endpoints_by_url,
        "memory": memory,
        "key_to_node_id": key_to_node_id,
        "trace_scorer": trace_scorer,
        "scorer": scorer,
        "retriever": retriever,
    }


def example_candidate_features(ctx: dict[str, Any], example: Any) -> np.ndarray:
    memory = ctx["memory"]
    key_to_node_id = ctx["key_to_node_id"]
    trace_scores, _ = ctx["trace_scorer"].candidate_scores(example, top_k=16)
    node_ids = [
        key_to_node_id.get(endpoint_key(candidate)) or key_to_node_id.get(str(candidate.get("api_name") or ""))
        for candidate in example.candidates
    ]
    valid_node_ids = [node_id for node_id in node_ids if node_id in memory.nodes]
    graph_scores = ctx["retriever"].score_nodes(step_text(example), "provider", valid_node_ids) if valid_node_ids else {}
    query = step_text(example)
    local_scores = local_text_scores(query, example.candidates)
    query_tokens = token_set(query)
    required_input_tokens = token_set(" ".join(example.required_inputs))
    task_domain = str(example.record_domain or "").lower()

    raw_rows: list[dict[str, float]] = []
    for idx, candidate in enumerate(example.candidates):
        node = memory.nodes.get(node_ids[idx] or "")
        params = required_param_names(candidate)
        param_tokens = token_set(" ".join(params))
        api_tokens = token_set(candidate.get("api_name"))
        text_tokens = token_set(candidate_text(candidate))
        default_tokens = token_set(
            " ".join(str(param.get("default") or "") for param in candidate.get("required_parameters") or [])
        )
        candidate_domain = str(candidate.get("domain") or "").lower()
        raw_rows.append(
            {
                "candidate_rank_reciprocal": 1.0 / float(idx + 1),
                "candidate_rank_index": float(idx + 1),
                "candidate_similarity": float(candidate.get("similarity_score") or 0.0),
                "local_bm25": float(local_scores["local_bm25"][idx]),
                "local_word_tfidf": float(local_scores["local_word_tfidf"][idx]),
                "local_char_tfidf": float(local_scores["local_char_tfidf"][idx]),
                "trace_match": max(
                    trace_scores.get(endpoint_key(candidate), 0.0),
                    trace_scores.get(str(candidate.get("api_name") or ""), 0.0),
                ),
                "graph_score": float(graph_scores.get(node_ids[idx] or "", 0.0)),
                "node_reliability": float(node.reliability) if node else 0.5,
                "node_risk": float(node.risk) if node else 0.0,
                "node_conflict": float(node.conflict) if node else 0.0,
                "param_name_overlap_with_required_inputs": overlap_ratio(param_tokens, required_input_tokens),
                "param_count": float(len(params)),
                "has_required_param": float(bool(params)),
                "same_domain_as_task": float(task_domain and (task_domain == candidate_domain or task_domain in candidate_domain)),
                "api_name_query_overlap": overlap_ratio(api_tokens, query_tokens),
                "candidate_text_query_overlap": overlap_ratio(text_tokens, query_tokens),
                "default_value_query_overlap": overlap_ratio(default_tokens, query_tokens),
            }
        )
    top_similarity = raw_rows[0]["candidate_similarity"] if raw_rows else 0.0
    top_bm25 = raw_rows[0]["local_bm25"] if raw_rows else 0.0
    top_word_tfidf = raw_rows[0]["local_word_tfidf"] if raw_rows else 0.0
    top_char_tfidf = raw_rows[0]["local_char_tfidf"] if raw_rows else 0.0
    top_trace = raw_rows[0]["trace_match"] if raw_rows else 0.0
    top_graph = raw_rows[0]["graph_score"] if raw_rows else 0.0
    mean_similarity = float(np.mean([row["candidate_similarity"] for row in raw_rows])) if raw_rows else 0.0
    rows: list[list[float]] = []
    for row in raw_rows:
        row["similarity_gap_to_top"] = row["candidate_similarity"] - top_similarity
        row["similarity_centered"] = row["candidate_similarity"] - mean_similarity
        row["local_bm25_gap_to_top"] = row["local_bm25"] - top_bm25
        row["local_word_tfidf_gap_to_top"] = row["local_word_tfidf"] - top_word_tfidf
        row["local_char_tfidf_gap_to_top"] = row["local_char_tfidf"] - top_char_tfidf
        row["trace_gap_to_top"] = row["trace_match"] - top_trace
        row["graph_gap_to_top"] = row["graph_score"] - top_graph
        rows.append([row[name] for name in FEATURE_NAMES])
    return np.asarray(rows, dtype=float)


def build_matrix(ctx: dict[str, Any], examples: list[Any]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    matrices: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[dict[str, Any]] = []
    offset = 0
    for example in examples:
        features = example_candidate_features(ctx, example)
        matrices.append(features)
        group_labels = np.zeros(len(example.candidates), dtype=int)
        group_labels[example.gold_index] = 1
        labels.extend(group_labels.tolist())
        groups.append(
            {
                "example": example,
                "offset": offset,
                "size": len(example.candidates),
                "gold_index": example.gold_index,
            }
        )
        offset += len(example.candidates)
    return np.vstack(matrices), np.asarray(labels, dtype=int), groups


def predict_scores(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    decision = model.decision_function(features)
    return np.asarray(decision, dtype=float)


def evaluate_grouped(
    model: Any,
    features: np.ndarray,
    groups: list[dict[str, Any]],
    ctx: dict[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    scores = predict_scores(model, features)
    elapsed = time.perf_counter() - start
    hits = 0
    top3 = 0
    mrr = 0.0
    workflows: dict[str, list[bool]] = {}
    para_f1: list[float] = []
    domain_hits: dict[str, list[int]] = {}
    feature_importance_rows: list[dict[str, float]] = []

    for group in groups:
        example = group["example"]
        begin = group["offset"]
        end = begin + group["size"]
        ranked = rank_from_scores(scores[begin:end].tolist())
        pred_index = ranked[0]
        correct = pred_index == group["gold_index"]
        hits += int(correct)
        top3 += int(group["gold_index"] in ranked[:3])
        mrr += 1.0 / (ranked.index(group["gold_index"]) + 1)
        workflows.setdefault(example.record_id, []).append(correct)
        domain_hits.setdefault(example.record_domain, []).append(int(correct))
        para_f1.append(
            f1_score(
                required_param_names(example.candidates[pred_index]),
                required_param_names(example.gold_candidate),
            )
        )

    n = len(groups)
    workflow_exact = sum(1 for values in workflows.values() if all(values)) / len(workflows) if workflows else 0.0
    return {
        "api_acc": hits / n if n else 0.0,
        "api_top3": top3 / n if n else 0.0,
        "api_mrr": mrr / n if n else 0.0,
        "workflow_exact": workflow_exact,
        "para_f1": float(np.mean(para_f1)) if para_f1 else 0.0,
        "avg_latency_ms_per_step": 1000.0 * elapsed / n if n else 0.0,
        "steps": n,
        "workflows": len(workflows),
        "domain_api_acc": {
            domain: sum(values) / len(values)
            for domain, values in sorted(domain_hits.items())
        },
    }


class FeatureColumnModel:
    def __init__(self, column_name: str) -> None:
        self.column_index = FEATURE_NAMES.index(column_name)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        score = features[:, self.column_index]
        score = (score - score.min()) / (score.max() - score.min() + 1e-9)
        return np.column_stack([1.0 - score, score])


def evaluate_hybrid_grouped(
    model: Any,
    features: np.ndarray,
    groups: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    scores = predict_scores(model, features)
    hits = 0
    top3 = 0
    mrr = 0.0
    workflows: dict[str, list[bool]] = {}
    para_f1: list[float] = []

    for group in groups:
        example = group["example"]
        begin = group["offset"]
        end = begin + group["size"]
        group_scores = scores[begin:end]
        model_ranked = sorted(range(len(group_scores)), key=lambda idx: (float(group_scores[idx]), -idx), reverse=True)
        semantic_index = 0
        model_index = model_ranked[0]
        margin = float(group_scores[model_index] - group_scores[semantic_index])
        pred_index = model_index if model_index != semantic_index and margin >= threshold else semantic_index
        correct = pred_index == group["gold_index"]
        hits += int(correct)
        # Hybrid produces a single top choice; use model ranking for top-3 diagnostic.
        top3 += int(group["gold_index"] in model_ranked[:3])
        if pred_index == group["gold_index"]:
            mrr += 1.0
        else:
            mrr += 1.0 / (model_ranked.index(group["gold_index"]) + 1)
        workflows.setdefault(example.record_id, []).append(correct)
        para_f1.append(
            f1_score(
                required_param_names(example.candidates[pred_index]),
                required_param_names(example.gold_candidate),
            )
        )

    n = len(groups)
    return {
        "api_acc": hits / n if n else 0.0,
        "api_top3": top3 / n if n else 0.0,
        "api_mrr": mrr / n if n else 0.0,
        "workflow_exact": sum(1 for values in workflows.values() if all(values)) / len(workflows) if workflows else 0.0,
        "para_f1": float(np.mean(para_f1)) if para_f1 else 0.0,
        "threshold": threshold,
        "steps": n,
        "workflows": len(workflows),
    }


def tune_hybrid_threshold(model: Any, features: np.ndarray, groups: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    thresholds = [round(value, 3) for value in np.linspace(0.0, 0.8, 41)]
    rows = [(threshold, evaluate_hybrid_grouped(model, features, groups, threshold)) for threshold in thresholds]
    threshold, metrics = max(rows, key=lambda item: (item[1]["api_acc"], item[1]["workflow_exact"], item[1]["api_mrr"]))
    return threshold, metrics


def feature_importance(model: Any) -> dict[str, float]:
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(estimator, "feature_importances_"):
        return {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, estimator.feature_importances_)
        }
    if hasattr(estimator, "coef_"):
        coef = estimator.coef_[0]
        return {name: float(value) for name, value in zip(FEATURE_NAMES, coef)}
    return {}


def train_models(seed: int) -> dict[str, Any]:
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=seed,
        ),
    }


def main() -> None:
    args = parse_args()
    ctx = build_context(args)
    matrices: dict[str, Any] = {}
    for split, examples in ctx["split_examples"].items():
        matrices[split] = build_matrix(ctx, examples)

    x_train, y_train, _ = matrices["train"]
    x_val, y_val, val_groups = matrices["val"]
    x_test, y_test, test_groups = matrices["test"]
    models = train_models(args.seed)
    results: dict[str, Any] = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        val_scores = predict_scores(model, x_val)
        test_scores = predict_scores(model, x_test)
        try:
            val_auc = float(roc_auc_score(y_val, val_scores))
            test_auc = float(roc_auc_score(y_test, test_scores))
        except ValueError:
            val_auc = 0.0
            test_auc = 0.0
        results[name] = {
            "val": evaluate_grouped(model, x_val, val_groups, ctx),
            "test": evaluate_grouped(model, x_test, test_groups, ctx),
            "val_auc": val_auc,
            "test_auc": test_auc,
            "feature_importance": feature_importance(model),
        }
        threshold, hybrid_val = tune_hybrid_threshold(model, x_val, val_groups)
        results[f"hybrid_semantic_{name}"] = {
            "val": hybrid_val,
            "test": evaluate_hybrid_grouped(model, x_test, test_groups, threshold),
            "base_model": name,
            "threshold": threshold,
            "feature_importance": feature_importance(model),
        }

    semantic_model = FeatureColumnModel("candidate_similarity")
    results["semantic_top1"] = {
        "val": evaluate_grouped(semantic_model, x_val, val_groups, ctx),
        "test": evaluate_grouped(semantic_model, x_test, test_groups, ctx),
        "feature_importance": {"candidate_similarity": 1.0},
    }

    best_name = max(results, key=lambda item: (results[item]["val"]["api_acc"], results[item]["val"]["api_mrr"]))
    payload = {
        "config": vars(args),
        "feature_names": FEATURE_NAMES,
        "dataset": {
            "train_steps": len(ctx["split_examples"]["train"]),
            "val_steps": len(ctx["split_examples"]["val"]),
            "test_steps": len(ctx["split_examples"]["test"]),
            "train_candidates": int(len(y_train)),
            "positive_rate": float(np.mean(y_train)),
        },
        "best_by_val_api_acc": best_name,
        "results": results,
        "notes": [
            "Models are trained only on train split gold API selections.",
            "Features exclude test-time observed_success labels.",
            "Reliability/risk features come from train-only GEMS memory and composition train traces.",
            "When contrastive_memory is enabled, non-gold train candidates receive weak label-negative feedback.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"best_by_val_api_acc: {best_name}")
    print(f"{'model':24s} {'val_acc':>8s} {'test_acc':>9s} {'test_wf':>8s} {'test_f1':>8s} {'test_mrr':>8s}")
    for name, row in results.items():
        print(
            f"{name:24s} {row['val']['api_acc']:8.4f} {row['test']['api_acc']:9.4f} "
            f"{row['test']['workflow_exact']:8.4f} {row['test']['para_f1']:8.4f} "
            f"{row['test']['api_mrr']:8.4f}"
        )
    print(f"saved {output}")


if __name__ == "__main__":
    main()
