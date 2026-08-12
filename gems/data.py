"""Load the processed RapidAPI-style service profile dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


JsonRecord = dict[str, Any]


def read_jsonl(path: str | Path) -> list[JsonRecord]:
    records: list[JsonRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
    return records


@dataclass
class ProcessedDataset:
    """In-memory view of ``dataset/processed``."""

    root: Path
    services: list[JsonRecord]
    endpoints: list[JsonRecord]
    feedback_events: list[JsonRecord]
    selection_tasks: list[JsonRecord]

    @classmethod
    def load(cls, root: str | Path) -> "ProcessedDataset":
        root = Path(root)
        return cls(
            root=root,
            services=read_jsonl(root / "services.jsonl"),
            endpoints=read_jsonl(root / "endpoints.jsonl"),
            feedback_events=read_jsonl(root / "feedback_events.jsonl"),
            selection_tasks=read_jsonl(root / "selection_tasks.jsonl"),
        )

    @property
    def services_by_id(self) -> dict[str, JsonRecord]:
        return {record["service_id"]: record for record in self.services}

    @property
    def endpoints_by_id(self) -> dict[str, JsonRecord]:
        return {record["endpoint_id"]: record for record in self.endpoints}

    @property
    def feedback_by_endpoint(self) -> dict[str, list[JsonRecord]]:
        grouped: dict[str, list[JsonRecord]] = {}
        for record in self.feedback_events:
            grouped.setdefault(record["endpoint_id"], []).append(record)
        return grouped

    def tasks(self, split: str | None = None) -> Iterable[JsonRecord]:
        for task in self.selection_tasks:
            if split is None or task.get("split") == split:
                yield task
