from __future__ import annotations

import unittest

from gems.evaluation import evaluate_endpoint_ranking
from gems.graph_memory import ExecutionGraphMemory
from gems.retrieval import RoleSpecificRetriever, has_reliability_intent, node_score


class GemsSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        services = [
            {
                "service_id": "svc_weather",
                "initial_confidence": 0.9,
                "qos": {"avg_latency": 100, "avg_service_level": 99, "avg_success_rate": 98},
            }
        ]
        endpoints = [
            {
                "endpoint_id": "ep_weather",
                "service_id": "svc_weather",
                "service_name": "Weather API",
                "endpoint_name": "Forecast",
                "endpoint_description": "Get weather forecast by city",
                "service_description": "Weather forecasts and current conditions",
                "interface_text": "GET /forecast required city",
                "category": "Weather",
                "method": "GET",
                "url": "https://weather.example/forecast",
                "observed_success": True,
                "required_parameters": [{"name": "city", "type": "STRING"}],
                "optional_parameters": [],
                "schema_summary": {"available": True, "fields": ["temperature"]},
                "split": "train",
            },
            {
                "endpoint_id": "ep_stock",
                "service_id": "svc_weather",
                "service_name": "Stock API",
                "endpoint_name": "Quote",
                "endpoint_description": "Get stock quote by ticker",
                "service_description": "Financial market data",
                "interface_text": "GET /quote required ticker",
                "category": "Finance",
                "method": "GET",
                "url": "https://stock.example/quote",
                "observed_success": True,
                "required_parameters": [{"name": "ticker", "type": "STRING"}],
                "optional_parameters": [],
                "schema_summary": {"available": True, "fields": ["price"]},
                "split": "train",
            },
        ]
        tasks = [
            {
                "task_id": "t1",
                "query": "Find a web API endpoint that can get weather forecast",
                "positive_endpoint_id": "ep_weather",
                "candidate_endpoint_ids": ["ep_weather", "ep_stock"],
                "positive_category": "Weather",
                "split": "test",
            }
        ]
        self.memory = ExecutionGraphMemory.from_processed_dataset(services, endpoints, [], tasks, train_only_tasks=False)
        self.tasks = tasks

    def test_retrieval_returns_api_evidence(self) -> None:
        retriever = RoleSpecificRetriever(self.memory)
        evidence = retriever.retrieve("weather forecast by city", "provider", top_k=5)
        api_rows = [row for row in evidence.serialized if row["kind"] == "node" and row["type"] == "api"]
        self.assertTrue(api_rows)

    def test_ranking_metrics(self) -> None:
        metrics, rows = evaluate_endpoint_ranking(self.memory, self.tasks)
        self.assertEqual(metrics.count, 1)
        self.assertEqual(rows[0]["rank"], 1)

    def test_endpoint_catalog_does_not_leak_outcome_label(self) -> None:
        endpoint = {
            "endpoint_id": "ep_failed",
            "endpoint_description": "A failed endpoint should not seed outcome features",
            "observed_success": False,
            "required_parameters": [],
            "optional_parameters": [],
            "schema_summary": {},
        }
        memory = ExecutionGraphMemory()
        memory.insert_endpoint(endpoint, {"initial_confidence": 0.8})
        node = memory.nodes["api:endpoint:ep_failed"]
        self.assertNotIn("observed_success", node.attrs)
        self.assertEqual(node.succ, 0)
        self.assertEqual(node.fail, 0)

    def test_reliability_intent_detection_is_provider_specific(self) -> None:
        self.assertTrue(has_reliability_intent("Find a reliable weather endpoint", "provider"))
        self.assertFalse(has_reliability_intent("Find a weather endpoint", "provider"))
        self.assertFalse(has_reliability_intent("Find a reliable weather endpoint", "planner"))

    def test_reliability_intent_score_promotes_reliable_candidate(self) -> None:
        semantic_top_score = node_score(
            similarity=0.95,
            reliability=0.60,
            type_prior=0.40,
            risk=0.0,
            conflict=0.0,
        )
        reliable_alt_score = node_score(
            similarity=0.30,
            reliability=0.80,
            type_prior=0.40,
            risk=0.0,
            conflict=0.0,
        )
        self.assertGreater(semantic_top_score, reliable_alt_score)

        reliability_intent_top_score = node_score(
            similarity=0.95,
            reliability=0.60,
            type_prior=0.40,
            risk=0.0,
            conflict=0.0,
            reliability_intent=True,
        )
        reliability_intent_alt_score = node_score(
            similarity=0.30,
            reliability=0.80,
            type_prior=0.40,
            risk=0.0,
            conflict=0.0,
            reliability_intent=True,
        )
        self.assertGreater(reliability_intent_alt_score, reliability_intent_top_score)


if __name__ == "__main__":
    unittest.main()
