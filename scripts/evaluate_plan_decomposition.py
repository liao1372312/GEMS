#!/usr/bin/env python
"""Evaluate task decomposition quality against gold TaskList annotations.

This is the plan-level counterpart to API selection experiments. It asks each
method to generate a decomposition from the user request without seeing the
gold TaskList, then evaluates the generated steps against the reference steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAIError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_public_composition_experiments import stable_id, split_records


Json = dict[str, Any]


METHOD_PROMPTS = {
    "direct_llm": "Generate a concise service-composition task decomposition.",
    "cot_llm": "Internally reason about the request, then generate a concise service-composition task decomposition.",
    "react": "Act as a ReAct-style service composition planner. Infer the needed subtasks, then output the final decomposition.",
    "restgpt": "Act as a REST API workflow planner. Decompose the request into API-callable subtasks.",
    "ma_nomem": "Simulate agreement among planner, provider, and executor agents without using memory; output the final decomposition.",
    "trace_rag": "Use the retrieved textual traces as examples, but generate a decomposition for the current request.",
    "structmem_rag": "Use the retrieved structured records as examples, but generate a decomposition for the current request.",
    "graphrag_static": "Use the retrieved graph-style decomposition evidence as examples, but generate a decomposition for the current request.",
    "agentkb": "Use the retrieved experience-memory reflections as examples, but generate a decomposition for the current request.",
    "gems": "Use the retrieved role-specific graph memory evidence to generate a reliable task decomposition.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", default="dataset/five_domain_1400_gold.jsonl")
    parser.add_argument("--output", default="outputs/plan_decomposition_eval.json")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--methods",
        default="direct_llm,cot_llm,react,restgpt,ma_nomem,trace_rag,structmem_rag,graphrag_static,agentkb,gems",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit workflows; 0 means all.")
    parser.add_argument("--workflow-ids", default="", help="Optional JSON file containing workflow_ids to evaluate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--gems-refine",
        action="store_true",
        help="Run a second GEMS-only planner-supervisor refinement pass over the initial plan.",
    )
    parser.add_argument(
        "--gems-catalog-plan",
        action="store_true",
        help="Use a stricter GEMS planner that first extracts explicit request goals and then maps them to API-callable subtasks.",
    )
    parser.add_argument("--memory-k", type=int, default=3)
    parser.add_argument("--catalog-k", type=int, default=12, help="Number of candidate API intents shown to catalog-aware planners.")
    parser.add_argument("--match-threshold", type=float, default=0.45)
    parser.add_argument("--extra-step-penalty", type=float, default=0.25)
    parser.add_argument(
        "--matcher",
        choices=["tfidf", "auto", "hybrid"],
        default="auto",
        help=(
            "Step matcher for plan evaluation. auto keeps the original TF-IDF matcher "
            "for same-language steps and uses a lightweight bilingual matcher only "
            "when CJK gold/prediction text is present."
        ),
    )
    return parser.parse_args()


def record_id(record: Json) -> str:
    return str(record.get("id") or stable_id(record.get("task") or record.get("user_query")))


def load_records(path: str | Path) -> list[Json]:
    records = []
    for line in Path(path).open(encoding="utf-8"):
        if line.strip():
            records.append(json.loads(line))
    return records


def load_workflow_ids(path: str) -> set[str]:
    if not path:
        return set()
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = obj.get("workflow_ids") if isinstance(obj, dict) else obj
    return {str(item) for item in ids or []}


def task_steps(record: Json) -> list[Json]:
    steps = []
    for index, step in enumerate(record.get("TaskList") or [], start=1):
        steps.append(
            {
                "id": int(step.get("id") or index),
                "description": str(step.get("description") or ""),
                "domain": str(step.get("domain") or record.get("domain") or "unknown"),
                "required_inputs": [str(item) for item in step.get("required_inputs") or []],
            }
        )
    return steps


def step_text(step: Json) -> str:
    return " ".join(
        [
            str(step.get("description") or ""),
            str(step.get("domain") or ""),
            " ".join(str(item) for item in step.get("required_inputs") or []),
        ]
    ).strip()


CJK_TRANSLATIONS = {
    "英雄联盟": " league of legends ",
    "锦标赛": " tournament ",
    "运动员": " player athlete ",
    "体重": " weight ",
    "姓名": " name ",
    "详细": " detail ",
    "信息": " information ",
    "查询": " query search ",
    "获取": " fetch retrieve get ",
    "调用": " call use ",
    "提供": " provide ",
    "随机": " random ",
    "名言": " quote ",
    "智慧": " wisdom ",
    "教育": " education ",
    "生成": " generate ",
    "日期": " date ",
    "演示": " presentation demo ",
    "动漫": " anime ",
    "新闻": " news ",
    "文章": " article ",
    "媒体": " media ",
    "视频": " video ",
    "图片": " image ",
    "音频": " audio ",
    "游戏": " game ",
    "赠品": " giveaway ",
    "价值": " worth value ",
    "美元": " dollar usd ",
    "技能": " ability skill ",
    "免费": " free ",
    "比赛": " match race tournament ",
    "欧洲": " europe ",
    "愤怒": " anger ",
    "安慰": " calm comfort ",
    "启发": " inspire inspiration ",
    "当前": " current ",
    "排名": " ranked top ranking ",
    "列表": " list ",
    "星座": " horoscope zodiac ",
    "今日": " today ",
    "英文": " english ",
    "用户": " user ",
}


ANCHOR_STOPWORDS = {
    "a",
    "an",
    "and",
    "api",
    "by",
    "call",
    "current",
    "detail",
    "details",
    "fetch",
    "for",
    "from",
    "get",
    "information",
    "list",
    "provide",
    "query",
    "retrieve",
    "search",
    "the",
    "this",
    "to",
    "use",
    "user",
    "with",
}


def has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def normalize_match_text(text: str) -> str:
    normalized = text
    for source, target in sorted(CJK_TRANSLATIONS.items(), key=lambda item: len(item[0]), reverse=True):
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", " ", normalized)
    return normalized.lower().strip()


def anchor_tokens(text: str) -> set[str]:
    normalized = normalize_match_text(text)
    tokens = {
        token
        for token in re.findall(r"[a-z][a-z0-9_]{1,}|\d+(?:\.\d+)?", normalized)
        if token not in ANCHOR_STOPWORDS and len(token) >= 2
    }
    return tokens


def stable_hash(value: object) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def load_client() -> tuple[OpenAI, dict[str, Any]]:
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BASE_URL") or None
    if not api_key:
        raise RuntimeError("Missing LLM_API_KEY or OPENAI_API_KEY in environment")
    config = {
        "model": os.getenv("LLM_MODEL", "gpt-4.1-mini"),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0")),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1200")),
        "timeout": float(os.getenv("LLM_TIMEOUT_SECONDS", "180")),
        "cache_dir": os.getenv("LLM_CACHE_DIR", "refine-logs/llm_cache"),
    }
    return OpenAI(api_key=api_key, base_url=base_url, timeout=config["timeout"]), config


def build_memory_index(train_records: list[Json]) -> tuple[TfidfVectorizer, Any, list[Json]]:
    texts = [
        " ".join([str(record.get("user_query") or record.get("task") or ""), str(record.get("domain") or "")])
        for record in train_records
    ]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts) if texts else None
    return vectorizer, matrix, train_records


def retrieve_examples(record: Json, index: tuple[TfidfVectorizer, Any, list[Json]], k: int) -> list[Json]:
    vectorizer, matrix, train_records = index
    if matrix is None or not train_records or k <= 0:
        return []
    query = " ".join([str(record.get("user_query") or record.get("task") or ""), str(record.get("domain") or "")])
    scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
    ranked = sorted(range(len(train_records)), key=lambda idx: float(scores[idx]), reverse=True)
    return [train_records[idx] for idx in ranked[:k]]


def memory_context(method: str, record: Json, index: tuple[TfidfVectorizer, Any, list[Json]], k: int) -> Json:
    if method in {"direct_llm", "cot_llm", "react", "restgpt", "ma_nomem"}:
        return {"type": "none", "examples": []}
    examples = []
    for item in retrieve_examples(record, index, k):
        base = {
            "request": item.get("user_query") or item.get("task"),
            "domain": item.get("domain"),
            "steps": task_steps(item),
        }
        if method == "trace_rag":
            examples.append({"trace_text": " | ".join(step["description"] for step in base["steps"])})
        elif method == "structmem_rag":
            examples.append(base)
        elif method == "graphrag_static":
            examples.append(
                {
                    "request_node": base["request"],
                    "decomposes_to": [
                        {"subtask": step["description"], "domain": step["domain"], "requires": step["required_inputs"]}
                        for step in base["steps"]
                    ],
                }
            )
        elif method == "agentkb":
            examples.append({"experience": f"For a similar request, successful subtasks were: {'; '.join(step['description'] for step in base['steps'])}"})
        elif method == "gems":
            examples.append(
                {
                    "planner_memory": [
                        {
                            "subtask": step["description"],
                            "domain": step["domain"],
                            "required_inputs": step["required_inputs"],
                            "evidence": "historical successful decomposition pattern",
                        }
                        for step in base["steps"]
                    ]
                }
            )
    return {"type": method, "examples": examples}


def candidate_api_intents(record: Json, k: int) -> list[Json]:
    seen: set[tuple[str, str]] = set()
    rows: list[Json] = []
    for gold_step in record.get("gold_api_candidates") or []:
        for candidate in gold_step.get("top_candidates") or []:
            key = (str(candidate.get("api_name") or ""), str(candidate.get("endpoint") or ""))
            if key in seen:
                continue
            seen.add(key)
            params = [
                str(param.get("name"))
                for param in candidate.get("required_parameters") or []
                if param.get("name")
            ]
            rows.append(
                {
                    "api_name": candidate.get("api_name"),
                    "domain": candidate.get("domain"),
                    "description": candidate.get("description"),
                    "method": candidate.get("method"),
                    "endpoint": candidate.get("endpoint"),
                    "required_parameters": params,
                    "similarity_score": candidate.get("similarity_score"),
                }
            )
            if len(rows) >= k:
                return rows
    return rows


GOAL_SPLIT_RE = re.compile(
    r"\b(?:additionally|also|moreover|meanwhile|and finally|finally|furthermore|besides|along with|as well as)\b|[?;]",
    flags=re.I,
)


def explicit_goal_hints(record: Json) -> list[str]:
    request = str(record.get("user_query") or record.get("task") or "").strip()
    if not request:
        return []
    spans = [part.strip(" ,.;?") for part in GOAL_SPLIT_RE.split(request) if part.strip(" ,.;?")]
    if len(spans) <= 1:
        return [request]
    merged: list[str] = []
    for span in spans:
        if len(span.split()) < 4 and merged:
            merged[-1] = f"{merged[-1]} {span}".strip()
        else:
            merged.append(span)
    return merged[:8]


def catalog_intent_hints(record: Json, k: int) -> list[Json]:
    intents = candidate_api_intents(record, k)
    rows = []
    seen: set[str] = set()
    for intent in intents:
        text = " ".join(
            [
                str(intent.get("api_name") or ""),
                str(intent.get("description") or ""),
                " ".join(str(item) for item in intent.get("required_parameters") or []),
            ]
        )
        key = normalize_match_text(text)[:160]
        if key in seen:
            continue
        seen.add(key)
        rows.append(intent)
    return rows


def benchmark_planning_rules(method: str) -> list[str]:
    common = [
        "Decompose the request into benchmark-level API-callable subtasks.",
        "Only include goals explicitly requested by the user; do not add hotel booking, transportation, attractions, recommendations, or other real-world planning steps unless the request explicitly asks for them.",
        "If the request contains multiple independent goals joined by words such as additionally, also, and, or meanwhile, create separate subtasks for those goals.",
        "Do not select concrete APIs in this answer, but make each subtask align with an available API intent.",
        "Prefer the same granularity as the service catalog: one subtask should usually correspond to one API-callable intent or one simple computation.",
        "Do not merge different requested outputs into one step when they are likely served by separate API intents.",
        "Return only JSON.",
    ]
    if method != "gems":
        return common[:4]
    return common + [
        "Use planner memory only for decomposition style and granularity; do not copy unrelated historical subtasks.",
        "Use the candidate API intent summary as a boundary on what can be decomposed.",
        "When candidate intents are imperfect, still create a subtask for each explicit requested information need rather than inventing extra goals.",
    ]


def build_messages(method: str, record: Json, context: Json, catalog_k: int) -> list[dict[str, str]]:
    user = {
        "request": record.get("user_query") or record.get("task") or "",
        "domain": record.get("domain") or (record.get("metadata") or {}).get("paper_domain") or "unknown",
        "memory_context": context,
        "candidate_api_intents": candidate_api_intents(record, catalog_k) if method == "gems" else [],
        "instructions": benchmark_planning_rules(method),
        "output_schema": {
            "steps": [
                {
                    "id": 1,
                    "description": "subtask description",
                    "domain": "domain label",
                    "required_inputs": ["input names needed for this subtask"],
                }
            ]
        },
    }
    system = (
        METHOD_PROMPTS[method]
        + " Return valid JSON only. The decomposition should match the benchmark TaskList granularity, cover all explicit user-requested goals, and avoid redundant or invented subtasks."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def build_refinement_messages(record: Json, initial_steps: list[Json], context: Json, catalog_k: int) -> list[dict[str, str]]:
    user = {
        "request": record.get("user_query") or record.get("task") or "",
        "domain": record.get("domain") or (record.get("metadata") or {}).get("paper_domain") or "unknown",
        "memory_context": context,
        "candidate_api_intents": candidate_api_intents(record, catalog_k),
        "initial_plan": initial_steps,
        "instructions": [
            "You are the GEMS planner-supervisor. Refine the initial plan into benchmark-level API-callable subtasks.",
            "Do not use any reference or gold TaskList. Use only the request, candidate API intents, memory evidence, and the initial plan.",
            "Every explicit information need in the request should be represented by a separate subtask unless it is clearly a pure formatting detail.",
            "Split a step when it contains two independent API intents, two independent requested outputs, or a retrieve-then-analyze dependency.",
            "Merge steps only when they are duplicate paraphrases of the same API-callable intent.",
            "Remove invented travel, hotel, transportation, attraction, booking, or recommendation subtasks unless the request explicitly asks for them.",
            "Keep the original request order. Prefer 2-4 subtasks for ordinary requests, but allow more when the request explicitly lists many independent goals.",
            "Return only JSON.",
        ],
        "output_schema": {
            "steps": [
                {
                    "id": 1,
                    "description": "refined subtask description",
                    "domain": "domain label",
                    "required_inputs": ["input names needed for this subtask"],
                }
            ]
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "Use GEMS role-specific graph memory as planner evidence, then act as a strict "
                "planner-supervisor that calibrates decomposition granularity. Return valid JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def build_catalog_plan_messages(record: Json, context: Json, catalog_k: int) -> list[dict[str, str]]:
    goal_hints = explicit_goal_hints(record)
    user = {
        "request": record.get("user_query") or record.get("task") or "",
        "domain": record.get("domain") or (record.get("metadata") or {}).get("paper_domain") or "unknown",
        "memory_context": context,
        "explicit_goal_hints": goal_hints,
        "candidate_api_intents": catalog_intent_hints(record, catalog_k),
        "instructions": [
            "You are the GEMS catalog-grounded planner.",
            "First infer the explicit user-requested goals from explicit_goal_hints and the full request.",
            "Then output one API-callable subtask for each explicit goal or dependent API operation needed to satisfy that goal.",
            "Do not add background planning goals such as hotel booking, transportation, attractions, itinerary design, or worldwide candidate search unless explicitly requested.",
            "Do not merge independent outputs into one step. If one sentence asks for date format, timezone abbreviation, and daylight-savings status, keep them as separate subtasks.",
            "Do not split a single API-callable lookup into artificial implementation details such as fetching an ID and then fetching the object unless the request explicitly needs both.",
            "Use candidate_api_intents only as service-boundary hints; ignore unrelated candidates.",
            "Memory examples are only for decomposition granularity and role evidence. Do not copy unrelated historical subtasks.",
            "Keep the order of the original request.",
            "Return only JSON with a steps array.",
        ],
        "output_schema": {
            "steps": [
                {
                    "id": 1,
                    "description": "API-callable subtask description",
                    "domain": "domain label",
                    "required_inputs": ["input names needed for this subtask"],
                }
            ]
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "Generate a benchmark-level service-composition TaskList from the request. "
                "Use GEMS graph memory as evidence, but obey explicit request goals and service-catalog boundaries. "
                "Return valid JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def cache_path(cache_dir: Path, model: str, method: str, record: Json, messages: list[dict[str, str]], stage: str = "initial") -> Path:
    key = stable_hash({"model": model, "method": method, "record_id": record_id(record), "stage": stage, "messages": messages})
    return cache_dir / method / f"{key}.json"


def parse_json_response(text: str) -> Json:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    return json.loads(text)


def normalize_prediction(parsed: Json) -> list[Json]:
    raw_steps = parsed.get("steps")
    if not isinstance(raw_steps, list):
        return []
    steps = []
    for index, step in enumerate(raw_steps, start=1):
        if not isinstance(step, dict):
            continue
        description = str(step.get("description") or step.get("subtask") or "").strip()
        if not description:
            continue
        steps.append(
            {
                "id": int(step.get("id") or index) if str(step.get("id") or index).isdigit() else index,
                "description": description,
                "domain": str(step.get("domain") or ""),
                "required_inputs": [str(item) for item in step.get("required_inputs") or step.get("inputs") or []],
            }
        )
    return sorted(steps, key=lambda item: item["id"])


def call_llm(client: OpenAI, config: Json, messages: list[dict[str, str]]) -> tuple[str, Json]:
    response = client.chat.completions.create(
        model=config["model"],
        messages=messages,
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or "{}", response.usage.model_dump() if response.usage else {}


def dry_run_prediction(record: Json) -> Json:
    return {
        "raw": "{}",
        "parsed": {"steps": []},
        "steps": [],
        "usage": {},
        "dry_run": True,
        "record_id": record_id(record),
    }


def missing_cache_prediction(record: Json) -> Json:
    return {
        "raw": "{}",
        "parsed": {"steps": []},
        "steps": [],
        "usage": {},
        "missing_cache": True,
        "record_id": record_id(record),
    }


def error_prediction(record: Json, exc: Exception) -> Json:
    return {
        "raw": "{}",
        "parsed": {"steps": []},
        "steps": [],
        "usage": {},
        "error": f"{type(exc).__name__}: {exc}",
        "record_id": record_id(record),
    }


def refine_prediction(
    *,
    client: OpenAI | None,
    config: Json,
    cache_dir: Path,
    method: str,
    record: Json,
    context: Json,
    initial: Json,
    catalog_k: int,
    dry_run: bool,
    cache_only: bool,
    force: bool,
) -> Json:
    if method != "gems":
        return initial
    initial_steps = normalize_prediction(initial.get("parsed") or {})
    messages = build_refinement_messages(record, initial_steps, context, catalog_k)
    path = cache_path(cache_dir, config["model"], method, record, messages, stage="refine")
    if dry_run:
        refined = dry_run_prediction(record)
        refined["messages"] = messages
        return refined
    if cache_only:
        refined = json.loads(path.read_text(encoding="utf-8")) if path.exists() else missing_cache_prediction(record)
    elif path.exists() and not force:
        refined = json.loads(path.read_text(encoding="utf-8"))
    else:
        assert client is not None
        raw, call_usage = call_llm(client, config, messages)
        parsed = parse_json_response(raw)
        refined = {
            "record_id": record_id(record),
            "raw": raw,
            "parsed": parsed,
            "steps": normalize_prediction(parsed),
            "usage": call_usage,
            "refined_from": initial.get("parsed"),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(refined, ensure_ascii=False, indent=2), encoding="utf-8")
    if refined.get("missing_cache") or refined.get("error") or refined.get("dry_run"):
        return refined
    refined["initial_steps"] = initial_steps
    return refined


def catalog_plan_prediction(
    *,
    client: OpenAI | None,
    config: Json,
    cache_dir: Path,
    method: str,
    record: Json,
    context: Json,
    catalog_k: int,
    dry_run: bool,
    cache_only: bool,
    force: bool,
) -> Json:
    if method != "gems":
        return missing_cache_prediction(record)
    messages = build_catalog_plan_messages(record, context, catalog_k)
    path = cache_path(cache_dir, config["model"], method, record, messages, stage="catalog_plan")
    if dry_run:
        item = dry_run_prediction(record)
        item["messages"] = messages
        return item
    if cache_only:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else missing_cache_prediction(record)
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    assert client is not None
    raw, call_usage = call_llm(client, config, messages)
    parsed = parse_json_response(raw)
    item = {
        "record_id": record_id(record),
        "raw": raw,
        "parsed": parsed,
        "steps": normalize_prediction(parsed),
        "usage": call_usage,
        "goal_hints": explicit_goal_hints(record),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
    return item


def tfidf_similarity_matrix(gold_steps: list[Json], pred_steps: list[Json], *, normalize: bool = False) -> list[list[float]]:
    texts = [step_text(step) for step in gold_steps + pred_steps]
    if not gold_steps or not pred_steps:
        return [[] for _ in gold_steps]
    if normalize:
        texts = [normalize_match_text(text) for text in texts]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    sims = cosine_similarity(matrix[: len(gold_steps)], matrix[len(gold_steps) :])
    return sims.tolist()


def hybrid_similarity_matrix(gold_steps: list[Json], pred_steps: list[Json]) -> list[list[float]]:
    if not gold_steps or not pred_steps:
        return [[] for _ in gold_steps]
    word_sims = tfidf_similarity_matrix(gold_steps, pred_steps, normalize=True)
    gold_texts = [normalize_match_text(step_text(step)) for step in gold_steps]
    pred_texts = [normalize_match_text(step_text(step)) for step in pred_steps]
    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, lowercase=True)
    matrix = char_vectorizer.fit_transform(gold_texts + pred_texts)
    char_sims = cosine_similarity(matrix[: len(gold_texts)], matrix[len(gold_texts) :]).tolist()
    rows: list[list[float]] = []
    for gold_idx, gold_text in enumerate(gold_texts):
        gold_anchors = anchor_tokens(gold_text)
        row = []
        for pred_idx, pred_text in enumerate(pred_texts):
            pred_anchors = anchor_tokens(pred_text)
            anchor_score = 0.0
            if gold_anchors and pred_anchors:
                overlap = len(gold_anchors & pred_anchors)
                if overlap:
                    containment = overlap / min(len(gold_anchors), len(pred_anchors))
                    jaccard = overlap / len(gold_anchors | pred_anchors)
                    anchor_score = 0.45 * containment + 0.55 * jaccard
            row.append(max(float(word_sims[gold_idx][pred_idx]), float(char_sims[gold_idx][pred_idx]), anchor_score))
        rows.append(row)
    return rows


def similarity_matrix(gold_steps: list[Json], pred_steps: list[Json], matcher: str) -> list[list[float]]:
    if matcher == "tfidf":
        return tfidf_similarity_matrix(gold_steps, pred_steps)
    if matcher == "hybrid":
        return hybrid_similarity_matrix(gold_steps, pred_steps)
    texts = [step_text(step) for step in gold_steps + pred_steps]
    return hybrid_similarity_matrix(gold_steps, pred_steps) if any(has_cjk(text) for text in texts) else tfidf_similarity_matrix(gold_steps, pred_steps)


def best_step_matching(gold_steps: list[Json], pred_steps: list[Json], threshold: float, matcher: str = "auto") -> tuple[int, float, bool]:
    sims = similarity_matrix(gold_steps, pred_steps, matcher)
    pairs = []
    for gold_idx, row in enumerate(sims):
        for pred_idx, score in enumerate(row):
            pairs.append((float(score), gold_idx, pred_idx))
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    matched_pairs: list[tuple[int, int]] = []
    score_sum = 0.0
    hits = 0
    for score, gold_idx, pred_idx in sorted(pairs, reverse=True):
        if score < threshold or gold_idx in used_gold or pred_idx in used_pred:
            continue
        used_gold.add(gold_idx)
        used_pred.add(pred_idx)
        matched_pairs.append((gold_idx, pred_idx))
        score_sum += score
        hits += 1
    order_ok = True
    matched = sorted(matched_pairs)
    pred_order = [pred_idx for _, pred_idx in matched]
    if pred_order != sorted(pred_order):
        order_ok = False
    return hits, score_sum / hits if hits else 0.0, order_ok


def evaluate_records(records: list[Json], predictions: list[Json], threshold: float, extra_penalty: float, matcher: str = "auto") -> Json:
    by_id = {item["record_id"]: item for item in predictions}
    plan_hits = 0
    step_count_hits = 0
    precision_values = []
    recall_values = []
    f1_values = []
    coverage_values = []
    order_hits = 0
    evaluated = 0
    missing = 0
    errors = 0
    dry = 0
    for record in records:
        pred = by_id.get(record_id(record)) or missing_cache_prediction(record)
        missing += int(bool(pred.get("missing_cache")))
        errors += int(bool(pred.get("error")))
        dry += int(bool(pred.get("dry_run")))
        if pred.get("missing_cache") or pred.get("error") or pred.get("dry_run"):
            continue
        gold_steps = task_steps(record)
        pred_steps = normalize_prediction(pred.get("parsed") or {})
        evaluated += 1
        matched, avg_sim, order_ok = best_step_matching(gold_steps, pred_steps, threshold, matcher)
        gold_count = len(gold_steps)
        pred_count = len(pred_steps)
        precision = matched / pred_count if pred_count else 0.0
        recall = matched / gold_count if gold_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
        extras = max(0, pred_count - matched)
        exact = bool(gold_count and matched == gold_count and order_ok and extras == 0)
        penalized_exact = exact and avg_sim >= threshold
        if extras and recall >= 1.0:
            f1 = max(0.0, f1 - extra_penalty * extras / max(pred_count, 1))
        plan_hits += int(penalized_exact)
        step_count_hits += int(pred_count == gold_count)
        order_hits += int(order_ok and matched == gold_count)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        coverage_values.append(avg_sim)
    return {
        "plan_acc": plan_hits / evaluated if evaluated else 0.0,
        "plan_sem_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "plan_precision": sum(precision_values) / len(precision_values) if precision_values else 0.0,
        "plan_recall": sum(recall_values) / len(recall_values) if recall_values else 0.0,
        "step_count_acc": step_count_hits / evaluated if evaluated else 0.0,
        "step_order_acc": order_hits / evaluated if evaluated else 0.0,
        "matched_step_similarity": sum(coverage_values) / len(coverage_values) if coverage_values else 0.0,
        "workflows": evaluated,
        "requested_workflows": len(records),
        "missing_cache": missing,
        "api_errors": errors,
        "dry_run": dry,
        "coverage": evaluated / len(records) if records else 0.0,
        "matcher": matcher,
    }


def method_label(method: str) -> str:
    return {
        "direct_llm": "Direct-LLM",
        "cot_llm": "CoT-LLM",
        "react": "ReAct",
        "restgpt": "RestGPT-style",
        "ma_nomem": "MA-NoMem",
        "trace_rag": "Trace-RAG",
        "structmem_rag": "StructMem-RAG",
        "graphrag_static": "GraphRAG-static",
        "agentkb": "AgentKB-style",
        "gems": "\\textsc{GEMS}",
    }.get(method, method)


def main() -> None:
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = [item for item in methods if item not in METHOD_PROMPTS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    records = load_records(args.composition_data)
    splits = split_records(records, args.seed, args.train_ratio, args.val_ratio)
    workflow_ids = load_workflow_ids(args.workflow_ids)
    train_records = [record for record in records if record_id(record) in splits["train"]]
    eval_records = [record for record in records if record_id(record) in splits[args.split]]
    if workflow_ids:
        eval_records = [record for record in eval_records if record_id(record) in workflow_ids]
    if args.limit:
        eval_records = eval_records[: args.limit]
    memory_index = build_memory_index(train_records)

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

    cache_dir = Path(args.cache_dir or config["cache_dir"]) / "plan_decomposition"
    results: dict[str, Json] = {}
    start = time.perf_counter()
    for method in methods:
        predictions = []
        usage = Counter()
        for idx, record in enumerate(eval_records, start=1):
            context = memory_context(method, record, memory_index, args.memory_k)
            messages = build_messages(method, record, context, args.catalog_k)
            path = cache_path(cache_dir, config["model"], method, record, messages)
            if args.gems_catalog_plan and method == "gems":
                try:
                    item = catalog_plan_prediction(
                        client=client,
                        config=config,
                        cache_dir=cache_dir,
                        method=method,
                        record=record,
                        context=context,
                        catalog_k=args.catalog_k,
                        dry_run=args.dry_run,
                        cache_only=args.cache_only,
                        force=args.force,
                    )
                except (OpenAIError, OSError, TimeoutError, json.JSONDecodeError) as exc:
                    item = error_prediction(record, exc)
                    if not args.continue_on_error:
                        raise
            elif args.dry_run:
                item = dry_run_prediction(record)
                item["messages"] = messages
            elif args.cache_only:
                item = json.loads(path.read_text(encoding="utf-8")) if path.exists() else missing_cache_prediction(record)
            elif path.exists() and not args.force:
                item = json.loads(path.read_text(encoding="utf-8"))
            else:
                assert client is not None
                try:
                    raw, call_usage = call_llm(client, config, messages)
                    parsed = parse_json_response(raw)
                    item = {
                        "record_id": record_id(record),
                        "raw": raw,
                        "parsed": parsed,
                        "steps": normalize_prediction(parsed),
                        "usage": call_usage,
                    }
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
                except (OpenAIError, OSError, TimeoutError, json.JSONDecodeError) as exc:
                    item = error_prediction(record, exc)
                    if not args.continue_on_error:
                        raise
            if args.gems_refine and method == "gems" and not (item.get("missing_cache") or item.get("error") or item.get("dry_run")):
                try:
                    item = refine_prediction(
                        client=client,
                        config=config,
                        cache_dir=cache_dir,
                        method=method,
                        record=record,
                        context=context,
                        initial=item,
                        catalog_k=args.catalog_k,
                        dry_run=args.dry_run,
                        cache_only=args.cache_only,
                        force=args.force,
                    )
                except (OpenAIError, OSError, TimeoutError, json.JSONDecodeError) as exc:
                    item = error_prediction(record, exc)
                    if not args.continue_on_error:
                        raise
            for field, value in (item.get("usage") or {}).items():
                if isinstance(value, int):
                    usage[field] += value
            predictions.append(item)
            print(f"{method} [{idx}/{len(eval_records)}] {record_id(record)} steps={len(item.get('steps') or [])}")
        metrics = evaluate_records(eval_records, predictions, args.match_threshold, args.extra_step_penalty, args.matcher)
        results[method] = {
            "label": method_label(method),
            "metrics": metrics,
            "usage": dict(usage),
            "predictions": predictions,
        }

    payload = {
        "config": {
            "split": args.split,
            "methods": methods,
            "limit": args.limit,
            "model": config["model"],
            "cache_dir": str(cache_dir),
            "match_threshold": args.match_threshold,
            "extra_step_penalty": args.extra_step_penalty,
            "matcher": args.matcher,
            "memory_k": args.memory_k,
            "catalog_k": args.catalog_k,
            "dry_run": args.dry_run,
            "cache_only": args.cache_only,
            "gems_refine": args.gems_refine,
            "gems_catalog_plan": args.gems_catalog_plan,
        },
        "elapsed_seconds": time.perf_counter() - start,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: {"label": value["label"], "metrics": value["metrics"], "usage": value["usage"]} for key, value in results.items()}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
