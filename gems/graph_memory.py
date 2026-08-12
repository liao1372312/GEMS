"""Execution-grounded typed graph memory for GEMS."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .text import compact_schema_text, lexical_key, normalize_text, sigmoid


NODE_TYPES = {
    "request",
    "subtask",
    "api",
    "parameter",
    "schema",
    "qos",
    "execution",
    "failure",
    "repair",
    "outcome",
}

EDGE_TYPES = {
    "decomposes_to",
    "selects",
    "binds",
    "requires",
    "produces",
    "depends_on",
    "causes",
    "repaired_by",
    "improves",
    "conflicts_with",
    "stale_under",
}


@dataclass
class Node:
    node_id: str
    node_type: str
    desc: str
    attrs: dict[str, Any] = field(default_factory=dict)
    succ: float = 0.0
    fail: float = 0.0
    reliability: float = 0.5
    risk: float = 0.0
    conflict: float = 0.0
    fresh: float = 1.0


@dataclass
class Edge:
    source: str
    target: str
    edge_type: str
    attrs: dict[str, Any] = field(default_factory=dict)
    reliability: float = 0.5


def _stable_hash(value: object, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def endpoint_api_node_id(endpoint_id: str) -> str:
    return f"api:endpoint:{endpoint_id}"


class ExecutionGraphMemory:
    """Typed graph memory with reliability propagation and outcome updates."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._adj_out: dict[str, list[int]] = {}
        self._adj_in: dict[str, list[int]] = {}

    def add_node(
        self,
        node_id: str,
        node_type: str,
        desc: str,
        attrs: dict[str, Any] | None = None,
        reliability: float | None = None,
    ) -> Node:
        if node_type not in NODE_TYPES:
            raise ValueError(f"Unknown node type: {node_type}")
        attrs = dict(attrs or {})
        if node_id in self.nodes:
            node = self.nodes[node_id]
            if desc and len(desc) > len(node.desc):
                node.desc = desc
            node.attrs.update({k: v for k, v in attrs.items() if v not in (None, "", [])})
            if reliability is not None:
                node.reliability = max(node.reliability, float(reliability))
            return node
        node = Node(
            node_id=node_id,
            node_type=node_type,
            desc=normalize_text(desc),
            attrs=attrs,
            reliability=float(0.5 if reliability is None else reliability),
        )
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        attrs: dict[str, Any] | None = None,
        reliability: float = 0.5,
    ) -> Edge:
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"Unknown edge type: {edge_type}")
        if source not in self.nodes or target not in self.nodes:
            raise KeyError(f"Missing edge endpoint: {source} -> {target}")
        edge = Edge(source, target, edge_type, dict(attrs or {}), float(reliability))
        index = len(self.edges)
        self.edges.append(edge)
        self._adj_out.setdefault(source, []).append(index)
        self._adj_in.setdefault(target, []).append(index)
        return edge

    def neighbors(self, node_id: str, include_incoming: bool = True) -> Iterable[tuple[str, Edge]]:
        for edge_index in self._adj_out.get(node_id, []):
            edge = self.edges[edge_index]
            yield edge.target, edge
        if include_incoming:
            for edge_index in self._adj_in.get(node_id, []):
                edge = self.edges[edge_index]
                yield edge.source, edge

    def node_texts(self) -> tuple[list[str], list[str]]:
        ids = list(self.nodes)
        docs = [self.nodes[node_id].desc for node_id in ids]
        return ids, docs

    def insert_endpoint(self, endpoint: dict[str, Any], service: dict[str, Any] | None = None) -> None:
        service = service or {}
        endpoint_id = endpoint["endpoint_id"]
        api_id = endpoint_api_node_id(endpoint_id)
        service_qos = service.get("qos") or {}
        desc_parts = [
            endpoint.get("interface_text"),
            endpoint.get("endpoint_description"),
            endpoint.get("service_description"),
            f"Category: {endpoint.get('category')}",
            f"Method: {endpoint.get('method')}",
        ]
        confidence = float(service.get("initial_confidence", 0.5) or 0.5)
        node = self.add_node(
            api_id,
            "api",
            ". ".join(normalize_text(part) for part in desc_parts if part),
            {
                "endpoint_id": endpoint_id,
                "service_id": endpoint.get("service_id"),
                "service_name": endpoint.get("service_name"),
                "endpoint_name": endpoint.get("endpoint_name"),
                "category": endpoint.get("category"),
                "method": endpoint.get("method"),
                "url": endpoint.get("url"),
                "split": endpoint.get("split"),
                "initial_confidence": confidence,
            },
            reliability=confidence,
        )

        for param in endpoint.get("required_parameters") or []:
            self._insert_parameter(api_id, endpoint_id, param, required=True)
        for param in endpoint.get("optional_parameters") or []:
            self._insert_parameter(api_id, endpoint_id, param, required=False)

        schema_summary = endpoint.get("schema_summary") or {}
        schema_id = f"schema:{endpoint_id}:{_stable_hash(schema_summary)}"
        schema_desc = compact_schema_text(schema_summary) or f"Schema for {endpoint.get('endpoint_name')}"
        self.add_node(schema_id, "schema", schema_desc, {"endpoint_id": endpoint_id, "schema": schema_summary})
        self.add_edge(api_id, schema_id, "produces", reliability=node.reliability)

        qos_id = f"qos:{endpoint_id}"
        qos_desc = (
            f"QoS for {endpoint.get('endpoint_name')}: latency {service_qos.get('avg_latency')}, "
            f"service level {service_qos.get('avg_service_level')}, "
            f"success rate {service_qos.get('avg_success_rate')}, popularity {service_qos.get('popularity_score')}."
        )
        self.add_node(qos_id, "qos", qos_desc, {"endpoint_id": endpoint_id, "qos": service_qos}, reliability=confidence)
        self.add_edge(api_id, qos_id, "produces", reliability=confidence)

    def _insert_parameter(self, api_id: str, endpoint_id: str, param: dict[str, Any], required: bool) -> None:
        name = normalize_text(param.get("name") or "unknown").lower()
        param_id = f"parameter:{endpoint_id}:{lexical_key(name, max_tokens=4)}"
        param_desc = (
            f"{'Required' if required else 'Optional'} parameter {name}. "
            f"Type: {param.get('type')}. Default: {param.get('default')}. "
            f"Description: {param.get('description')}."
        )
        self.add_node(
            param_id,
            "parameter",
            param_desc,
            {"endpoint_id": endpoint_id, "name": name, "required": required, "raw": param},
        )
        self.add_edge(api_id, param_id, "requires" if required else "binds", {"required": required})

    def insert_feedback_event(self, event: dict[str, Any]) -> None:
        endpoint_id = event["endpoint_id"]
        api_id = endpoint_api_node_id(endpoint_id)
        if api_id not in self.nodes:
            return
        event_id = event["event_id"]
        execution_id = f"execution:{event_id}"
        success = bool(event.get("success"))
        event_desc = (
            f"Execution feedback for endpoint {endpoint_id}. "
            f"Success: {success}. Status: {event.get('statuscode')}. "
            f"Latency: {event.get('latency_ms')} ms. Response: {event.get('response_summary')}."
        )
        rel = 0.85 if success else 0.25
        self.add_node(execution_id, "execution", event_desc, event, reliability=rel)
        self.add_edge(api_id, execution_id, "depends_on", reliability=rel)

        outcome_id = f"outcome:{event_id}"
        outcome_desc = "Successful endpoint execution." if success else "Failed endpoint execution."
        self.add_node(outcome_id, "outcome", outcome_desc, {"success": success}, reliability=rel)
        self.add_edge(execution_id, outcome_id, "produces", reliability=rel)
        self.apply_outcome_credit([api_id, execution_id, outcome_id], success=success)

        if not success:
            failure_id = f"failure:{event_id}"
            summary = event.get("response_summary") or {}
            failure_desc = f"Failure for endpoint {endpoint_id}. Status: {event.get('statuscode')}. Summary: {summary}."
            self.add_node(failure_id, "failure", failure_desc, event, reliability=0.75)
            self.add_edge(failure_id, execution_id, "causes", reliability=0.75)
            self.nodes[api_id].risk = min(1.0, self.nodes[api_id].risk + 0.15)

    def insert_selection_task(self, task: dict[str, Any], endpoint: dict[str, Any] | None = None) -> None:
        task_id = task["task_id"]
        request_id = f"request:{task_id}"
        subtask_id = f"subtask:{task_id}"
        query = normalize_text(task.get("query"))
        self.add_node(request_id, "request", query, task, reliability=0.7)
        self.add_node(subtask_id, "subtask", query, {"task_id": task_id, "category": task.get("positive_category")})
        self.add_edge(request_id, subtask_id, "decomposes_to", reliability=0.7)

        api_id = endpoint_api_node_id(task["positive_endpoint_id"])
        if api_id in self.nodes:
            self.add_edge(subtask_id, api_id, "selects", {"weak_supervision": True}, reliability=0.65)
            if endpoint:
                success = endpoint.get("observed_success")
                if success is not None:
                    self.apply_outcome_credit([request_id, subtask_id, api_id], success=bool(success), eta=0.05)

    def apply_outcome_credit(
        self,
        node_ids: Iterable[str],
        success: bool,
        eta: float = 0.12,
        decay: float = 0.01,
        stale: float = 0.0,
        conflict: float = 0.0,
    ) -> None:
        credit = 1.0 if success else -1.0
        for node_id in node_ids:
            node = self.nodes.get(node_id)
            if not node:
                continue
            if success:
                node.succ += 1
            else:
                node.fail += 1
                node.risk = min(1.0, node.risk + 0.1)
            node.reliability = float(
                min(1.0, max(0.0, (1.0 - decay) * node.reliability + eta * credit - stale - conflict))
            )

    def add_conflict(self, first_node: str, second_node: str, reason: str) -> None:
        if first_node not in self.nodes or second_node not in self.nodes:
            return
        self.add_edge(first_node, second_node, "conflicts_with", {"reason": reason}, reliability=0.8)
        self.nodes[first_node].conflict = min(1.0, self.nodes[first_node].conflict + 0.2)
        self.nodes[second_node].conflict = min(1.0, self.nodes[second_node].conflict + 0.2)

    def mark_stale(self, node_id: str, catalog_reason: str) -> None:
        node = self.nodes.get(node_id)
        if not node:
            return
        stale_id = f"failure:stale:{_stable_hash([node_id, catalog_reason])}"
        self.add_node(stale_id, "failure", f"Stale memory marker: {catalog_reason}", {"reason": catalog_reason})
        self.add_edge(node_id, stale_id, "stale_under", {"reason": catalog_reason}, reliability=0.8)
        node.fresh = max(0.0, node.fresh - 0.4)
        node.risk = min(1.0, node.risk + 0.25)

    def propagate_reliability(self, layers: int = 2) -> None:
        """Run deterministic typed propagation inspired by the paper equations."""

        node_ids = list(self.nodes)
        if not node_ids:
            return
        idx = {node_id: i for i, node_id in enumerate(node_ids)}
        features = np.vstack([self._initial_features(self.nodes[node_id]) for node_id in node_ids])
        h = features.astype(float)
        relation_weights = {
            "decomposes_to": 0.80,
            "selects": 1.15,
            "binds": 0.85,
            "requires": 0.90,
            "produces": 1.00,
            "depends_on": 1.05,
            "causes": -0.80,
            "repaired_by": 0.75,
            "improves": 0.90,
            "conflicts_with": -0.65,
            "stale_under": -0.75,
        }
        type_bias = {
            "request": 0.05,
            "subtask": 0.08,
            "api": 0.12,
            "parameter": 0.02,
            "schema": 0.02,
            "qos": 0.08,
            "execution": 0.05,
            "failure": -0.15,
            "repair": 0.07,
            "outcome": 0.10,
        }
        degrees = np.ones(len(node_ids))
        for edge in self.edges:
            if edge.source in idx and edge.target in idx:
                degrees[idx[edge.source]] += 1
                degrees[idx[edge.target]] += 1

        for _ in range(max(0, layers)):
            messages = np.zeros_like(h)
            for edge in self.edges:
                if edge.source not in idx or edge.target not in idx:
                    continue
                src = idx[edge.source]
                dst = idx[edge.target]
                norm = math.sqrt(degrees[src] * degrees[dst])
                weight = relation_weights.get(edge.edge_type, 0.5) * edge.reliability / norm
                messages[dst] += weight * h[src]
                messages[src] += 0.25 * weight * h[dst]
            h = np.tanh(0.65 * h + messages)

        for node_id, vector in zip(node_ids, h):
            node = self.nodes[node_id]
            logit = (
                1.8 * vector[0]
                - 1.3 * vector[1]
                + 0.9 * vector[2]
                - 1.0 * vector[3]
                - 1.2 * vector[4]
                + 0.7 * vector[5]
                + type_bias.get(node.node_type, 0.0)
            )
            node.reliability = min(1.0, max(0.0, 0.35 * node.reliability + 0.65 * sigmoid(float(logit))))
            node.risk = min(1.0, max(0.0, 0.6 * node.risk + 0.4 * float(vector[1] + vector[3]) / 2.0))
            node.conflict = min(1.0, max(0.0, 0.7 * node.conflict + 0.3 * float(vector[4])))

    def _initial_features(self, node: Node) -> np.ndarray:
        total = node.succ + node.fail
        success_rate = node.succ / total if total else node.reliability
        failure_prior = node.fail / total if total else node.risk
        repair_success = float(node.attrs.get("repair_success_rate", 0.0) or 0.0)
        drift_score = 1.0 - float(node.fresh)
        conflict_score = float(node.conflict)
        recency_score = float(node.attrs.get("recency", 1.0) or 1.0)
        return np.array([success_rate, failure_prior, repair_success, drift_score, conflict_score, recency_score])

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(node) for node in self.nodes.values()],
            "edges": [asdict(edge) for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExecutionGraphMemory":
        memory = cls()
        for node_payload in payload.get("nodes", []):
            node = Node(**node_payload)
            memory.nodes[node.node_id] = node
        for edge_payload in payload.get("edges", []):
            edge = Edge(**edge_payload)
            index = len(memory.edges)
            memory.edges.append(edge)
            memory._adj_out.setdefault(edge.source, []).append(index)
            memory._adj_in.setdefault(edge.target, []).append(index)
        return memory

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ExecutionGraphMemory":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_processed_dataset(
        cls,
        services: list[dict[str, Any]],
        endpoints: list[dict[str, Any]],
        feedback_events: list[dict[str, Any]],
        selection_tasks: list[dict[str, Any]] | None = None,
        train_only_tasks: bool = True,
    ) -> "ExecutionGraphMemory":
        memory = cls()
        services_by_id = {record["service_id"]: record for record in services}
        endpoints_by_id = {record["endpoint_id"]: record for record in endpoints}
        for endpoint in endpoints:
            memory.insert_endpoint(endpoint, services_by_id.get(endpoint.get("service_id")))
        for event in feedback_events:
            memory.insert_feedback_event(event)
        for task in selection_tasks or []:
            if train_only_tasks and task.get("split") != "train":
                continue
            endpoint = endpoints_by_id.get(task["positive_endpoint_id"])
            memory.insert_selection_task(task, endpoint)
        memory.propagate_reliability(layers=2)
        return memory
