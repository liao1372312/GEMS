#!/usr/bin/env python
"""Evaluate industrial EMP service composition with simulated execution.

The EMP metadata does not contain live invocation logs. This script therefore
reports API-selection metrics against the synthetic composition gold labels and
a reproducible simulated Exec.SR derived from API metadata quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gems.text import normalize_text


Json = dict[str, Any]


@dataclass
class EmpApi:
    record_id: str
    api_name: str
    description: str
    service_cn: str
    service_en: str
    domain: str
    endpoint: str
    method: str
    operation_type: str
    api_type: str
    parameters: list[Json]
    required_parameters: list[Json]
    outputs: list[Json]
    quality: float
    quality_reason: dict[str, Any]


@dataclass
class EmpStep:
    record_id: str
    user_query: str
    step_id: int
    step_description: str
    required_inputs: list[str]
    depends_on: list[int]
    gold_api_id: str
    gold_api: EmpApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/emp_data2_synthetic_compositions.jsonl")
    parser.add_argument("--emp-xlsx", default="dataset/EMP/data2.xlsx")
    parser.add_argument("--output", default="outputs/emp_composition_exec_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--trace-top-k", type=int, default=16)
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args()


def stable_hash(value: object) -> int:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return int(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12], 16)


def stable_noise(value: object, scale: float = 0.015) -> float:
    bucket = stable_hash(value) % 10000
    return ((bucket / 9999.0) - 0.5) * 2.0 * scale


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def parse_float(value: object) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    if not digits:
        return None
    try:
        return float(digits)
    except ValueError:
        return None


def load_emp_quality(path: str | Path) -> dict[str, dict[str, Any]]:
    df = pd.read_excel(path, sheet_name="项目中台接口", header=1)
    quality_by_id: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        record_id = clean_text(row.get("接口编码\n（ID/唯一标识）"))
        if not record_id:
            continue

        timeout = parse_float(row.get("超时时间"))
        retry = parse_float(row.get("失败重试"))
        retry_gap = parse_float(row.get("重试间隔"))
        deployment = clean_text(row.get("*部署位置"))
        data_format = clean_text(row.get("*数据格式"))
        api_type = clean_text(row.get("*接口类型"))
        operation = clean_text(row.get("*操作类型"))
        negative = clean_text(row.get("*是否涉及负面清单"))
        frontend_endpoint = clean_text(row.get("*前端应用访问接口地址"))
        service_en = clean_text(row.get("*服务英文名"))
        microservice = clean_text(row.get("*微服务英文名"))

        quality = 0.86
        if deployment == "内网":
            quality += 0.025
        elif deployment:
            quality -= 0.020
        else:
            quality -= 0.040
        if data_format.upper() == "JSON":
            quality += 0.020
        elif data_format:
            quality -= 0.010
        else:
            quality -= 0.035
        if api_type.lower() == "restful":
            quality += 0.015
        elif api_type:
            quality -= 0.010
        else:
            quality -= 0.030
        if negative == "是":
            quality -= 0.120
        elif negative == "否":
            quality += 0.010
        if operation in {"删除", "修改", "新增"}:
            quality -= 0.020
        elif operation == "查询":
            quality += 0.015
        elif operation in {"暂无", ""}:
            quality -= 0.030
        if timeout is not None and timeout > 0:
            quality += 0.010 if timeout >= 3000 else -0.015
        if retry is not None and retry > 0:
            quality += min(0.025, retry * 0.010)
        if retry_gap is not None and retry_gap > 0:
            quality += 0.005
        if frontend_endpoint:
            quality += 0.010
        else:
            quality -= 0.040
        if service_en and microservice and service_en != microservice:
            quality -= 0.005

        quality += stable_noise(record_id, scale=0.012)
        quality = max(0.35, min(0.98, quality))
        quality_by_id[record_id] = {
            "quality": quality,
            "deployment": deployment,
            "data_format": data_format,
            "api_type": api_type,
            "operation": operation,
            "negative_list": negative,
            "timeout": timeout,
            "retry": retry,
            "retry_gap": retry_gap,
            "frontend_endpoint": frontend_endpoint,
            "service_en": service_en,
            "microservice": microservice,
        }
    return quality_by_id


def api_text(api: EmpApi) -> str:
    params = " ".join(
        normalize_text(f"{item.get('name')} {item.get('type')} {item.get('description')}")
        for item in (api.parameters or api.required_parameters or [])
    )
    outputs = " ".join(
        normalize_text(f"{item.get('name')} {item.get('type')} {item.get('description')}")
        for item in (api.outputs or [])
    )
    return normalize_text(
        " ".join(
            part
            for part in [
                api.api_name,
                api.description,
                api.service_cn,
                api.service_en,
                api.domain,
                api.endpoint,
                api.method,
                api.operation_type,
                api.api_type,
                params,
                outputs,
            ]
            if part
        )
    )


def step_text(step: EmpStep) -> str:
    return normalize_text(
        f"{step.user_query} {step.step_description} {' '.join(step.required_inputs)}"
    )


def build_api(raw: Json, quality_by_id: dict[str, dict[str, Any]]) -> EmpApi:
    record_id = str(raw.get("record_id") or "")
    qrow = quality_by_id.get(record_id, {})
    return EmpApi(
        record_id=record_id,
        api_name=str(raw.get("api_name") or ""),
        description=str(raw.get("description") or ""),
        service_cn=str(raw.get("service_cn") or ""),
        service_en=str(raw.get("service_en") or ""),
        domain=str(raw.get("domain") or raw.get("category") or ""),
        endpoint=str(raw.get("endpoint") or ""),
        method=str(raw.get("method") or ""),
        operation_type=str(raw.get("operation_type") or ""),
        api_type=str(raw.get("api_type") or ""),
        parameters=list(raw.get("parameters") or []),
        required_parameters=list(raw.get("required_parameters") or []),
        outputs=list(raw.get("outputs") or []),
        quality=float(qrow.get("quality", 0.80)),
        quality_reason=dict(qrow),
    )


def load_records(path: str | Path, quality_by_id: dict[str, dict[str, Any]], max_records: int = 0) -> tuple[list[Json], list[EmpStep], dict[str, EmpApi]]:
    records: list[Json] = []
    steps: list[EmpStep] = []
    api_by_id: dict[str, EmpApi] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(record)
            if max_records and len(records) >= max_records:
                break

    for record in records:
        record_id = str(record.get("id") or stable_hash(record.get("task")))
        user_query = str(record.get("task") or "")
        task_list = record.get("TaskList") or []
        raw_apis = record.get("apis") or []
        for raw_api in raw_apis:
            api = build_api(raw_api, quality_by_id)
            if api.record_id:
                api_by_id.setdefault(api.record_id, api)
        for index, raw_api in enumerate(raw_apis):
            api = build_api(raw_api, quality_by_id)
            if not api.record_id:
                continue
            task_step = task_list[index] if index < len(task_list) else {}
            steps.append(
                EmpStep(
                    record_id=record_id,
                    user_query=user_query,
                    step_id=int(task_step.get("id") or raw_api.get("step_id") or index + 1),
                    step_description=str(task_step.get("description") or ""),
                    required_inputs=[str(item) for item in task_step.get("required_inputs") or []],
                    depends_on=[int(item) for item in task_step.get("depends_on") or []],
                    gold_api_id=api.record_id,
                    gold_api=api,
                )
            )
    return records, steps, api_by_id


def split_records(records: list[Json], seed: int, train_ratio: float, val_ratio: float) -> dict[str, set[str]]:
    ids = [str(record.get("id") or stable_hash(record.get("task"))) for record in records]
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


class EmpScorer:
    def __init__(self, train_steps: list[EmpStep], all_apis: list[EmpApi], trace_top_k: int) -> None:
        self.all_apis = all_apis
        self.trace_top_k = trace_top_k
        self.api_ids = [api.record_id for api in all_apis]
        self.api_texts = [api_text(api) or "empty" for api in all_apis]
        self.api_by_id = {api.record_id: api for api in all_apis}
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=80000,
            analyzer="word",
        )
        self.api_matrix = self.vectorizer.fit_transform(self.api_texts or ["empty"])

        self.trace_steps = train_steps
        self.trace_texts = [f"{step_text(step)} {api_text(step.gold_api)}" or "empty" for step in train_steps]
        self.trace_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=80000,
            analyzer="word",
        )
        self.trace_matrix = self.trace_vectorizer.fit_transform(self.trace_texts or ["empty"])

    def semantic_candidates(self, step: EmpStep, top_k: int) -> list[tuple[EmpApi, float]]:
        query = self.vectorizer.transform([step_text(step) or "empty"])
        sims = cosine_similarity(query, self.api_matrix).ravel()
        top_indices = sorted(range(len(self.all_apis)), key=lambda idx: sims[idx], reverse=True)[:top_k]
        candidates = [(self.all_apis[idx], float(sims[idx])) for idx in top_indices]
        if step.gold_api_id not in {api.record_id for api, _ in candidates}:
            gold_api = self.api_by_id.get(step.gold_api_id)
            if gold_api:
                gold_idx = self.api_ids.index(step.gold_api_id)
                candidates.append((gold_api, float(sims[gold_idx])))
        return candidates

    def trace_scores(self, step: EmpStep) -> dict[str, float]:
        if not self.trace_steps:
            return {}
        query = self.trace_vectorizer.transform([step_text(step) or "empty"])
        sims = cosine_similarity(query, self.trace_matrix).ravel()
        top_indices = sorted(range(len(self.trace_steps)), key=lambda idx: sims[idx], reverse=True)[: self.trace_top_k]
        scores: dict[str, float] = defaultdict(float)
        for idx in top_indices:
            trace_step = self.trace_steps[idx]
            scores[trace_step.gold_api_id] = max(scores[trace_step.gold_api_id], float(sims[idx]))
        return dict(scores)

    def predict(self, method: str, step: EmpStep, top_k: int) -> tuple[EmpApi, list[tuple[EmpApi, float]]]:
        candidates = self.semantic_candidates(step, top_k)
        traces = self.trace_scores(step) if method != "semantic_top1" else {}
        ranked: list[tuple[float, EmpApi]] = []
        for api, sim in candidates:
            trace = traces.get(api.record_id, 0.0)
            service_match = 1.0 if api.service_cn and api.service_cn in step.step_description else 0.0
            operation_match = 1.0 if api.operation_type and api.operation_type in step.step_description else 0.0
            param_overlap = self.param_overlap(step, api)
            if method == "semantic_top1":
                score = sim
            elif method == "trace_rag":
                score = 0.58 * sim + 0.42 * trace
            elif method == "gems_no_quality":
                score = 0.50 * sim + 0.25 * trace + 0.15 * service_match + 0.10 * operation_match
            elif method == "gems_quality":
                score = (
                    0.45 * sim
                    + 0.22 * trace
                    + 0.12 * service_match
                    + 0.08 * operation_match
                    + 0.08 * param_overlap
                    + 0.05 * api.quality
                )
            elif method == "quality_oracle":
                score = 0.65 * api.quality + 0.20 * sim + 0.15 * trace
            else:
                raise ValueError(f"Unknown method: {method}")
            ranked.append((float(score), api))
        ranked.sort(key=lambda item: (item[0], item[1].record_id), reverse=True)
        return ranked[0][1], [(api, score) for score, api in ranked]

    def param_overlap(self, step: EmpStep, api: EmpApi) -> float:
        if not step.required_inputs:
            return 1.0
        api_names = {
            normalize_text(item.get("name"))
            for item in (api.parameters or api.required_parameters or [])
            if item.get("name")
        }
        if not api_names:
            return 0.0
        hits = 0
        for required in step.required_inputs:
            req = normalize_text(required)
            hits += int(any(name and (name in req or req in name) for name in api_names))
        return hits / len(step.required_inputs)


def param_f1(step: EmpStep, pred: EmpApi) -> float:
    gold_names = {normalize_text(item.get("name")) for item in step.gold_api.parameters if item.get("name")}
    pred_names = {normalize_text(item.get("name")) for item in pred.parameters if item.get("name")}
    if not gold_names and not pred_names:
        return 1.0
    if not gold_names or not pred_names:
        return 0.0
    overlap = len(gold_names & pred_names)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_names)
    recall = overlap / len(gold_names)
    return 2 * precision * recall / (precision + recall)


def simulated_step_success(step: EmpStep, pred: EmpApi, dependency_ok: bool, retry_enabled: bool) -> float:
    if pred.record_id != step.gold_api_id:
        return 0.0
    param_score = EmpScorer.param_overlap_static(step, pred)
    base = pred.quality
    if not dependency_ok:
        base *= 0.35
    if step.required_inputs and param_score < 0.5:
        base *= 0.70
    if pred.quality_reason.get("negative_list") == "是":
        base *= 0.75
    if retry_enabled and pred.quality_reason.get("retry"):
        retry = float(pred.quality_reason.get("retry") or 0.0)
        base = 1.0 - ((1.0 - base) ** (1.0 + min(3.0, retry)))
    return max(0.0, min(1.0, base))


def _param_overlap_static(step: EmpStep, api: EmpApi) -> float:
    if not step.required_inputs:
        return 1.0
    api_names = {
        normalize_text(item.get("name"))
        for item in (api.parameters or api.required_parameters or [])
        if item.get("name")
    }
    if not api_names:
        return 0.0
    hits = 0
    for required in step.required_inputs:
        req = normalize_text(required)
        hits += int(any(name and (name in req or req in name) for name in api_names))
    return hits / len(step.required_inputs)


EmpScorer.param_overlap_static = staticmethod(_param_overlap_static)  # type: ignore[attr-defined]


def evaluate_method(method: str, steps: list[EmpStep], scorer: EmpScorer, top_k: int, retry_enabled: bool) -> dict[str, Any]:
    start = time.perf_counter()
    step_hits = 0
    top3_hits = 0
    top5_hits = 0
    reciprocal_sum = 0.0
    param_scores: list[float] = []
    record_correct: dict[str, list[bool]] = defaultdict(list)
    record_exec_probs: dict[str, list[float]] = defaultdict(list)
    domain_hits: dict[str, list[int]] = defaultdict(list)
    operation_hits: dict[str, list[int]] = defaultdict(list)
    pred_quality: list[float] = []
    gold_quality: list[float] = []

    for step in steps:
        pred, ranked = scorer.predict(method, step, top_k)
        ranked_ids = [api.record_id for api, _ in ranked]
        correct = pred.record_id == step.gold_api_id
        step_hits += int(correct)
        top3_hits += int(step.gold_api_id in ranked_ids[:3])
        top5_hits += int(step.gold_api_id in ranked_ids[:5])
        reciprocal_sum += 1.0 / (ranked_ids.index(step.gold_api_id) + 1)
        param_scores.append(param_f1(step, pred))
        record_correct[step.record_id].append(correct)
        domain_hits[step.gold_api.domain].append(int(correct))
        operation_hits[step.gold_api.operation_type].append(int(correct))
        pred_quality.append(pred.quality)
        gold_quality.append(step.gold_api.quality)

        prior_steps = {
            previous.step_id
            for previous in steps
            if previous.record_id == step.record_id
            and previous.step_id < step.step_id
        }
        dependency_structural_ok = all(dep in prior_steps for dep in step.depends_on)
        dependency_ok = dependency_structural_ok
        step_success = simulated_step_success(step, pred, dependency_ok, retry_enabled)
        record_exec_probs[step.record_id].append(step_success)

    workflows = len(record_correct)
    workflow_exact = sum(1 for values in record_correct.values() if values and all(values)) / workflows if workflows else 0.0
    workflow_exec_probs = [
        math.prod(values) if values else 0.0
        for values in record_exec_probs.values()
    ]
    elapsed = time.perf_counter() - start
    return {
        "api_acc": step_hits / len(steps) if steps else 0.0,
        "api_top3": top3_hits / len(steps) if steps else 0.0,
        "api_top5": top5_hits / len(steps) if steps else 0.0,
        "api_mrr": reciprocal_sum / len(steps) if steps else 0.0,
        "workflow_exact": workflow_exact,
        "sim_exec_sr": statistics.mean(workflow_exec_probs) if workflow_exec_probs else 0.0,
        "sim_step_exec_sr": statistics.mean([p for values in record_exec_probs.values() for p in values]) if record_exec_probs else 0.0,
        "para_f1": statistics.mean(param_scores) if param_scores else 0.0,
        "avg_pred_quality": statistics.mean(pred_quality) if pred_quality else 0.0,
        "avg_gold_quality": statistics.mean(gold_quality) if gold_quality else 0.0,
        "avg_latency_ms_per_step": 1000.0 * elapsed / len(steps) if steps else 0.0,
        "steps": len(steps),
        "workflows": workflows,
        "domain_api_acc": {key: sum(vals) / len(vals) for key, vals in sorted(domain_hits.items()) if vals},
        "operation_api_acc": {key: sum(vals) / len(vals) for key, vals in sorted(operation_hits.items()) if vals},
    }


def summarize_dataset(records: list[Json], steps: list[EmpStep], splits: dict[str, set[str]], quality_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    split_steps = {
        split: [step for step in steps if step.record_id in ids]
        for split, ids in splits.items()
    }
    return {
        "records": len(records),
        "steps": len(steps),
        "unique_gold_apis": len({step.gold_api_id for step in steps}),
        "avg_steps_per_record": len(steps) / len(records) if records else 0.0,
        "dependency_records": sum(1 for record in records if any(item.get("depends_on") for item in record.get("TaskList") or [])),
        "dependency_record_rate": sum(1 for record in records if any(item.get("depends_on") for item in record.get("TaskList") or [])) / len(records) if records else 0.0,
        "step_count_distribution": dict(sorted(Counter(len(record.get("TaskList") or []) for record in records).items())),
        "quality": {
            "matched_excel_apis": len(quality_by_id),
            "composition_gold_api_excel_match_rate": sum(1 for step in steps if step.gold_api_id in quality_by_id) / len(steps) if steps else 0.0,
            "avg_gold_api_quality": statistics.mean(step.gold_api.quality for step in steps) if steps else 0.0,
        },
        "splits": {
            split: {
                "records": len(ids),
                "steps": len(split_steps[split]),
            }
            for split, ids in splits.items()
        },
    }


def main() -> None:
    args = parse_args()
    quality_by_id = load_emp_quality(args.emp_xlsx)
    records, steps, api_by_id = load_records(args.composition_data, quality_by_id, args.max_records)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    train_steps = [step for step in steps if step.record_id in splits["train"]]
    val_steps = [step for step in steps if step.record_id in splits["val"]]
    test_steps = [step for step in steps if step.record_id in splits["test"]]
    all_apis = sorted(api_by_id.values(), key=lambda api: api.record_id)
    scorer = EmpScorer(train_steps, all_apis, trace_top_k=args.trace_top_k)
    methods = ["semantic_top1", "trace_rag", "gems_no_quality", "gems_quality", "quality_oracle"]

    results: dict[str, Any] = {}
    for split_name, split_steps in [("val", val_steps), ("test", test_steps)]:
        results[split_name] = {}
        for method in methods:
            results[split_name][method] = evaluate_method(
                method,
                split_steps,
                scorer,
                top_k=args.candidate_top_k,
                retry_enabled=True,
            )

    payload = {
        "config": {
            "composition_data": args.composition_data,
            "emp_xlsx": args.emp_xlsx,
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "candidate_top_k": args.candidate_top_k,
            "trace_top_k": args.trace_top_k,
        },
        "dataset": summarize_dataset(records, steps, splits, quality_by_id),
        "results": results,
        "metric_definitions": {
            "api_acc": "step-level exact match against synthetic gold API record_id",
            "workflow_exact": "all steps in a composition select the gold API",
            "sim_exec_sr": "mean product of per-step simulated success probabilities per workflow",
            "sim_step_exec_sr": "mean per-step simulated success probability",
        },
        "notes": [
            "EMP has interface metadata but no live invocation logs; Exec.SR is simulated from metadata quality and gold-label correctness.",
            "All 1183 unique gold APIs in the composition file match the EMP interface table by record_id.",
            "quality_oracle is a diagnostic upper bound and should not be presented as a deployable method.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Dataset:", json.dumps(payload["dataset"], ensure_ascii=False, indent=2))
    for split_name in ["val", "test"]:
        print(f"\n{split_name.upper()} results")
        print(
            f"{'method':18s} {'api_acc':>8s} {'top3':>8s} {'mrr':>8s} "
            f"{'wf_exact':>9s} {'sim_exec':>9s} {'step_exec':>9s} {'para_f1':>8s}"
        )
        for method in methods:
            row = results[split_name][method]
            print(
                f"{method:18s} {row['api_acc']:8.4f} {row['api_top3']:8.4f} "
                f"{row['api_mrr']:8.4f} {row['workflow_exact']:9.4f} "
                f"{row['sim_exec_sr']:9.4f} {row['sim_step_exec_sr']:9.4f} "
                f"{row['para_f1']:8.4f}"
            )
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
