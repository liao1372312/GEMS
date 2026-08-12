from __future__ import annotations

import unittest

from scripts.evaluate_plan_decomposition import evaluate_records, explicit_goal_hints


class PlanDecompositionEvalTest(unittest.TestCase):
    def test_exact_decomposition_counts_as_plan_hit(self) -> None:
        records = [
            {
                "id": "r1",
                "TaskList": [
                    {"id": 1, "description": "Find current weather by city", "domain": "weather", "required_inputs": ["city"]},
                    {"id": 2, "description": "Summarize weather conditions", "domain": "weather", "required_inputs": ["weather"]},
                ],
            }
        ]
        predictions = [
            {
                "record_id": "r1",
                "parsed": {
                    "steps": [
                        {"id": 1, "description": "Find current weather by city", "domain": "weather", "required_inputs": ["city"]},
                        {"id": 2, "description": "Summarize weather conditions", "domain": "weather", "required_inputs": ["weather"]},
                    ]
                },
            }
        ]
        metrics = evaluate_records(records, predictions, threshold=0.45, extra_penalty=0.25)
        self.assertEqual(metrics["plan_acc"], 1.0)
        self.assertEqual(metrics["step_count_acc"], 1.0)
        self.assertEqual(metrics["plan_sem_f1"], 1.0)

    def test_extra_step_blocks_exact_plan_accuracy(self) -> None:
        records = [
            {
                "id": "r1",
                "TaskList": [
                    {"id": 1, "description": "Find current weather by city", "domain": "weather", "required_inputs": ["city"]},
                ],
            }
        ]
        predictions = [
            {
                "record_id": "r1",
                "parsed": {
                    "steps": [
                        {"id": 1, "description": "Find current weather by city", "domain": "weather", "required_inputs": ["city"]},
                        {"id": 2, "description": "Book a hotel", "domain": "travel", "required_inputs": ["city"]},
                    ]
                },
            }
        ]
        metrics = evaluate_records(records, predictions, threshold=0.45, extra_penalty=0.25)
        self.assertEqual(metrics["plan_acc"], 0.0)
        self.assertLess(metrics["plan_sem_f1"], 1.0)

    def test_auto_matcher_handles_bilingual_gold_steps(self) -> None:
        records = [
            {
                "id": "r1",
                "TaskList": [
                    {"id": 1, "description": "调用API获取一条关于教育的随机名言。", "domain": "cross_domain", "required_inputs": []},
                    {"id": 2, "description": "调用API生成一个2022年内的随机日期，用于演示。", "domain": "cross_domain", "required_inputs": []},
                ],
            }
        ]
        predictions = [
            {
                "record_id": "r1",
                "parsed": {
                    "steps": [
                        {"id": 1, "description": "Get a random quote about education", "domain": "cross_domain", "required_inputs": []},
                        {
                            "id": 2,
                            "description": "Generate a random date between 2022-01-01 and 2022-12-31 for a presentation",
                            "domain": "cross_domain",
                            "required_inputs": [],
                        },
                    ]
                },
            }
        ]
        metrics = evaluate_records(records, predictions, threshold=0.45, extra_penalty=0.25, matcher="auto")
        self.assertEqual(metrics["plan_acc"], 1.0)
        self.assertEqual(metrics["plan_sem_f1"], 1.0)

    def test_explicit_goal_hints_split_multi_goal_request(self) -> None:
        record = {
            "user_query": (
                "Fetch current time in London. Additionally, get the timezone abbreviation. "
                "Also check whether it is daylight savings."
            )
        }
        hints = explicit_goal_hints(record)
        self.assertGreaterEqual(len(hints), 3)
        self.assertIn("timezone", " ".join(hints).lower())


if __name__ == "__main__":
    unittest.main()
