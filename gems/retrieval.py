"""Reliability-aware role-specific subgraph retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .graph_memory import ExecutionGraphMemory, Node
from .text import TfidfTextIndex, normalize_text


ROLE_DESCRIPTIONS = {
    "planner": "task decomposition patterns and similar user requests",
    "provider": "API selection evidence, schema compatibility, QoS, outcomes, service reliability",
    "executor": "parameter binding, required inputs, output schemas, data dependencies",
    "supervisor": "execution failures, repairs, conflicts, stale evidence, final outcomes",
}

ROLE_TYPE_PRIORS = {
    "planner": {"request": 0.35, "subtask": 0.35, "api": 0.10, "outcome": 0.05},
    "provider": {"api": 0.40, "schema": 0.15, "qos": 0.20, "outcome": 0.15, "failure": -0.10},
    "executor": {"parameter": 0.35, "schema": 0.25, "api": 0.15, "execution": 0.10},
    "supervisor": {"failure": 0.35, "repair": 0.30, "outcome": 0.20, "api": 0.10, "qos": 0.05},
}

ROLE_TYPE_MASKS = {
    "planner": {"request", "subtask", "api", "outcome"},
    "provider": {"api", "schema", "qos", "execution", "failure", "outcome"},
    "executor": {"api", "parameter", "schema", "execution", "outcome"},
    "supervisor": {"api", "failure", "repair", "outcome", "qos", "execution"},
}

DEFAULT_SCORE_WEIGHTS = {
    "similarity": 0.48,
    "reliability": 0.32,
    "type_prior": 0.16,
    "risk": 0.12,
    "conflict": 0.10,
}

RELIABILITY_INTENT_SCORE_WEIGHTS = {
    "similarity": 0.0,
    "reliability": 1.0,
    "type_prior": 0.04,
    "risk": 0.50,
    "conflict": 0.20,
}

RELIABILITY_INTENT_TERMS = {
    "reliable",
    "reliability",
    "robust",
    "stable",
    "safe",
    "successful",
    "trustworthy",
    "dependable",
    "risk",
    "risky",
    "failure",
    "failed",
    "avoid failure",
    "avoid failed",
    "not fail",
}


@dataclass
class RetrievedEvidence:
    role: str
    query: str
    node_ids: list[str]
    scores: dict[str, float]
    serialized: list[dict[str, Any]]


def has_reliability_intent(query: str, role: str) -> bool:
    """Detect queries where reliability should dominate semantic closeness."""
    if role.lower() != "provider":
        return False
    normalized = normalize_text(query).lower()
    padded = f" {normalized} "
    return any(f" {term} " in padded for term in RELIABILITY_INTENT_TERMS)


def node_score(
    *,
    similarity: float,
    reliability: float,
    type_prior: float,
    risk: float,
    conflict: float,
    reliability_intent: bool = False,
) -> float:
    weights = RELIABILITY_INTENT_SCORE_WEIGHTS if reliability_intent else DEFAULT_SCORE_WEIGHTS
    return (
        weights["similarity"] * similarity
        + weights["reliability"] * reliability
        + weights["type_prior"] * type_prior
        - weights["risk"] * risk
        - weights["conflict"] * conflict
    )


class RoleSpecificRetriever:
    def __init__(self, memory: ExecutionGraphMemory) -> None:
        self.memory = memory
        node_ids, docs = memory.node_texts()
        self.text_index = TfidfTextIndex.fit(node_ids, docs)

    def role_query(self, query: str, role: str) -> str:
        return f"{normalize_text(query)} {ROLE_DESCRIPTIONS.get(role, role)}"

    def retrieve(
        self,
        query: str,
        role: str,
        seed_top_k: int = 20,
        hops: int = 1,
        top_k: int = 12,
    ) -> RetrievedEvidence:
        role = role.lower()
        q = self.role_query(query, role)
        similarities = self.text_index.similarities(q)
        seed_ids = [
            node_id
            for node_id, _ in sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:seed_top_k]
        ]
        candidates = self._expand(seed_ids, hops=hops)
        mask = ROLE_TYPE_MASKS.get(role)
        if mask:
            candidates = {node_id for node_id in candidates if self.memory.nodes[node_id].node_type in mask}
        scores = self.score_nodes(query, role, candidates, similarities=similarities)
        ranked = [node_id for node_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]]
        return RetrievedEvidence(
            role=role,
            query=query,
            node_ids=ranked,
            scores={node_id: scores[node_id] for node_id in ranked},
            serialized=self.serialize(ranked, scores),
        )

    def score_nodes(
        self,
        query: str,
        role: str,
        node_ids: Iterable[str],
        similarities: dict[str, float] | None = None,
    ) -> dict[str, float]:
        role = role.lower()
        q = self.role_query(query, role)
        node_ids = list(node_ids)
        similarities = similarities or self.text_index.score_subset(q, node_ids)
        priors = ROLE_TYPE_PRIORS.get(role, {})
        reliability_intent = has_reliability_intent(query, role)
        scores: dict[str, float] = {}
        for node_id in node_ids:
            node = self.memory.nodes.get(node_id)
            if not node:
                continue
            type_prior = priors.get(node.node_type, 0.0)
            scores[node_id] = node_score(
                similarity=similarities.get(node_id, 0.0),
                reliability=node.reliability,
                type_prior=type_prior,
                risk=node.risk,
                conflict=node.conflict,
                reliability_intent=reliability_intent,
            )
        return scores

    def serialize(self, node_ids: Iterable[str], scores: dict[str, float]) -> list[dict[str, Any]]:
        selected = set(node_ids)
        rows: list[dict[str, Any]] = []
        for node_id in node_ids:
            node = self.memory.nodes[node_id]
            rows.append(
                {
                    "kind": "node",
                    "id": node_id,
                    "type": node.node_type,
                    "score": round(scores.get(node_id, 0.0), 4),
                    "reliability": round(node.reliability, 4),
                    "risk": round(node.risk, 4),
                    "description": node.desc[:500],
                    "attrs": self._compact_attrs(node),
                }
            )
        for edge in self.memory.edges:
            if edge.source in selected and edge.target in selected:
                rows.append(
                    {
                        "kind": "edge",
                        "source": edge.source,
                        "relation": edge.edge_type,
                        "target": edge.target,
                        "reliability": round(edge.reliability, 4),
                        "attrs": edge.attrs,
                    }
                )
        return rows

    def _expand(self, seed_ids: Iterable[str], hops: int) -> set[str]:
        seen = set(seed_ids)
        frontier = set(seed_ids)
        for _ in range(max(0, hops)):
            next_frontier: set[str] = set()
            for node_id in frontier:
                for neighbor_id, _ in self.memory.neighbors(node_id):
                    if neighbor_id not in seen:
                        seen.add(neighbor_id)
                        next_frontier.add(neighbor_id)
            frontier = next_frontier
        return seen

    def _compact_attrs(self, node: Node) -> dict[str, Any]:
        keep = [
            "endpoint_id",
            "service_id",
            "service_name",
            "endpoint_name",
            "category",
            "method",
            "url",
            "success",
            "required",
        ]
        return {key: node.attrs[key] for key in keep if key in node.attrs}
