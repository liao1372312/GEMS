#!/usr/bin/env python
"""Train a field-aware reranker optimized for plan accuracy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gems.text import normalize_text
from run_public_composition_experiments import (
    candidate_text,
    f1_score,
    load_composition_examples,
    required_param_names,
    split_records,
    step_text,
)


FEATURE_NAMES = [
    "rank_recip",
    "rank_index",
    "sim",
    "sim_gap_top",
    "word_tfidf",
    "char_tfidf",
    "bm25",
    "api_name_overlap",
    "description_overlap",
    "endpoint_overlap",
    "domain_overlap",
    "param_required_overlap",
    "param_query_overlap",
    "param_count",
    "has_params",
    "method_get",
    "method_post",
    "same_step_domain",
    "query_api_exact_action",
    "query_endpoint_action",
]


ACTION_TERMS = {
    "get": {"get", "fetch", "retrieve", "query", "search", "list", "find", "lookup", "check", "details"},
    "create": {"create", "add", "new", "insert", "post"},
    "update": {"update", "modify", "edit", "change"},
    "delete": {"delete", "remove"},
    "convert": {"convert", "translate", "transform"},
    "send": {"send", "email", "message", "notify"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--output", default="outputs/field_aware_plan_reranker.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def tokens(text: object) -> set[str]:
    return {
        token
        for token in normalize_text(text).replace("/", " ").replace("-", " ").replace("_", " ").split()
        if token
    }


def overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def action_match(query_tokens: set[str], text: object) -> float:
    text_tokens = tokens(text)
    for action_words in ACTION_TERMS.values():
        if query_tokens & action_words and text_tokens & action_words:
            return 1.0
    return 0.0


def local_scores(example: Any) -> dict[str, list[float]]:
    query = step_text(example)
    docs = [candidate_text(candidate) or "empty" for candidate in example.candidates]
    tokenized_docs = [doc.split() for doc in docs]
    query_tokens = query.split()
    bm25 = BM25Okapi(tokenized_docs).get_scores(query_tokens).astype(float).tolist() if query_tokens else [0.0] * len(docs)
    if bm25:
        lo, hi = min(bm25), max(bm25)
        bm25 = [(value - lo) / (hi - lo + 1e-9) for value in bm25]

    word = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    word_matrix = word.fit_transform(docs + [query or "empty"])
    word_scores = cosine_similarity(word_matrix[-1], word_matrix[:-1]).ravel().astype(float).tolist()

    char = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    char_matrix = char.fit_transform(docs + [query or "empty"])
    char_scores = cosine_similarity(char_matrix[-1], char_matrix[:-1]).ravel().astype(float).tolist()
    return {"bm25": bm25, "word_tfidf": word_scores, "char_tfidf": char_scores}


def example_features(example: Any) -> np.ndarray:
    query = step_text(example)
    query_tokens = tokens(query)
    required_tokens = tokens(" ".join(example.required_inputs))
    step_domain_tokens = tokens(example.record_domain)
    scores = local_scores(example)
    top_sim = float(example.candidates[0].get("similarity_score") or 0.0)
    rows = []
    for idx, candidate in enumerate(example.candidates):
        params = required_param_names(candidate)
        param_tokens = tokens(" ".join(params))
        method = str(candidate.get("method") or "").upper()
        api_name = candidate.get("api_name")
        endpoint = candidate.get("endpoint")
        description = candidate.get("description")
        domain = candidate.get("domain")
        rows.append(
            [
                1.0 / float(idx + 1),
                float(idx + 1),
                float(candidate.get("similarity_score") or 0.0),
                float(candidate.get("similarity_score") or 0.0) - top_sim,
                scores["word_tfidf"][idx],
                scores["char_tfidf"][idx],
                scores["bm25"][idx],
                overlap(query_tokens, tokens(api_name)),
                overlap(query_tokens, tokens(description)),
                overlap(query_tokens, tokens(endpoint)),
                overlap(query_tokens, tokens(domain)),
                overlap(required_tokens, param_tokens),
                overlap(query_tokens, param_tokens),
                float(len(params)),
                float(bool(params)),
                float(method == "GET"),
                float(method == "POST"),
                overlap(step_domain_tokens, tokens(domain)),
                action_match(query_tokens, api_name),
                action_match(query_tokens, endpoint),
            ]
        )
    return np.asarray(rows, dtype=float)


def build_matrix(examples: list[Any]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    xs = []
    ys = []
    groups = []
    offset = 0
    for example in examples:
        feats = example_features(example)
        xs.append(feats)
        labels = np.zeros(len(example.candidates), dtype=int)
        labels[example.gold_index] = 1
        ys.extend(labels.tolist())
        groups.append({"example": example, "offset": offset, "size": len(example.candidates)})
        offset += len(example.candidates)
    return np.vstack(xs), np.asarray(ys, dtype=int), groups


def score_model(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def evaluate_scores(scores: np.ndarray, groups: list[dict[str, Any]], alpha: float, threshold: float) -> dict[str, Any]:
    hits = 0
    top3 = 0
    mrr = 0.0
    workflows: dict[str, list[bool]] = {}
    para = []
    changed = 0
    for group in groups:
        example = group["example"]
        begin = group["offset"]
        end = begin + group["size"]
        model_scores = scores[begin:end]
        semantic = np.asarray([float(c.get("similarity_score") or 0.0) for c in example.candidates])
        combined = alpha * model_scores + (1.0 - alpha) * semantic
        ranked = sorted(range(len(combined)), key=lambda idx: (float(combined[idx]), -idx), reverse=True)
        best = ranked[0]
        # Conservative plan mode: keep semantic top-1 unless the reranker is
        # sufficiently more confident than candidate 1.
        if best != 0 and (combined[best] - combined[0]) < threshold:
            best = 0
        changed += int(best != 0)
        correct = best == example.gold_index
        hits += int(correct)
        top3 += int(example.gold_index in ranked[:3])
        mrr += 1.0 / (ranked.index(example.gold_index) + 1)
        workflows.setdefault(example.record_id, []).append(correct)
        para.append(f1_score(required_param_names(example.candidates[best]), required_param_names(example.gold_candidate)))
    n = len(groups)
    return {
        "api_acc": hits / n if n else 0.0,
        "api_top3": top3 / n if n else 0.0,
        "api_mrr": mrr / n if n else 0.0,
        "workflow_exact": sum(1 for values in workflows.values() if all(values)) / len(workflows) if workflows else 0.0,
        "para_f1": sum(para) / len(para) if para else 0.0,
        "change_rate": changed / n if n else 0.0,
        "steps": n,
        "workflows": len(workflows),
    }


def main() -> None:
    args = parse_args()
    records, examples = load_composition_examples(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    split_examples = {split: [e for e in examples if e.record_id in ids] for split, ids in splits.items()}
    x_train, y_train, _ = build_matrix(split_examples["train"])
    x_val, _, val_groups = build_matrix(split_examples["val"])
    x_test, _, test_groups = build_matrix(split_examples["test"])

    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)),
        "rf": RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=args.seed, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=400, max_depth=9, min_samples_leaf=4, class_weight="balanced", random_state=args.seed, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(max_iter=160, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.04, random_state=args.seed),
    }
    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        val_scores = score_model(model, x_val)
        test_scores = score_model(model, x_test)
        for alpha in [0.2, 0.35, 0.5, 0.65, 0.8, 1.0]:
            for threshold in [-0.02, -0.01, 0.0, 0.005, 0.01, 0.02, 0.05]:
                val = evaluate_scores(val_scores, val_groups, alpha, threshold)
                rows.append(
                    {
                        "name": name,
                        "alpha": alpha,
                        "threshold": threshold,
                        "val": val,
                        "test_scores": test_scores,
                    }
                )
    rows.sort(key=lambda row: (row["val"]["workflow_exact"], row["val"]["api_acc"], row["val"]["para_f1"]), reverse=True)
    best = rows[0]
    test = evaluate_scores(best["test_scores"], test_groups, best["alpha"], best["threshold"])
    semantic = evaluate_scores(np.zeros(len(x_test)), test_groups, 0.0, 1.0)
    payload = {
        "config": vars(args),
        "feature_names": FEATURE_NAMES,
        "best": {k: v for k, v in best.items() if k != "test_scores"},
        "field_aware_reranker_test": test,
        "semantic_top1_test": semantic,
        "top_10": [{k: v for k, v in row.items() if k != "test_scores"} for row in rows[:10]],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
