# GEMS Implementation

This repository contains a lightweight implementation of the paper method in
`ICSOC_GNN/samplepaper_gnn.tex`: execution-grounded graph memory for reliable,
role-specific service composition evidence.

## Main Pieces

- `gems/graph_memory.py`: typed nodes and edges, execution feedback insertion,
  outcome-driven reliability updates, and deterministic typed propagation.
- `gems/retrieval.py`: planner/provider/executor/supervisor role-specific
  subgraph retrieval with reliability, risk, conflict, and type-prior scoring.
- `scripts/build_gems_memory.py`: builds `outputs/gems_memory.json`.
- `scripts/evaluate_retrieval.py`: evaluates provider endpoint ranking on
  `selection_tasks.jsonl`.
- `scripts/inspect_gems_evidence.py`: prints serialized evidence for a query.
- `scripts/export_gems_subgraph.py`: exports retrieved role-specific subgraphs
  as HTML, GraphML, JSON, and PNG.

## Quick Start

```bash
python scripts/build_gems_memory.py --data-dir dataset/processed --output outputs/gems_memory.json
python scripts/evaluate_retrieval.py --memory outputs/gems_memory.json --split test
python scripts/inspect_gems_evidence.py "Find an API for weather forecast by city" --memory outputs/gems_memory.json --role provider
python scripts/export_gems_subgraph.py "Find an API for weather forecast by city" --memory outputs/gems_memory.json --role provider --output-prefix outputs/weather_provider_subgraph
```

By default, memory construction uses all service and endpoint catalog entries,
but only `train` feedback events and `train` weak-supervision task traces as
historical execution evidence. Use `--feedback-splits all` only for diagnostic
or transductive analysis.

The implementation intentionally keeps the first runnable version dependency
light: `numpy` and `scikit-learn` are enough for reliability propagation and
semantic retrieval over the provided processed dataset.
