# Experiment Code Review

Date: 2026-07-04

Scope: local-only review of the first GEMS implementation for
`ICSOC_GNN/samplepaper.tex`.

## Method Coverage

- Execution-grounded graph memory is implemented in `gems/graph_memory.py` with
  typed request, subtask, API, parameter, schema, QoS, execution, failure, and
  outcome nodes.
- Typed relations include decomposition, API selection, parameter/schema
  evidence, execution dependencies, failure causes, conflict, and stale markers.
- Reliability propagation follows the paper structure with deterministic
  typed-message passing over node features: success rate, failure prior, repair
  rate, drift, conflict, and recency.
- Outcome-driven updates adjust node reliability from observed execution
  success/failure and increase risk after failures.
- Role-specific retrieval supports planner, provider, executor, and supervisor
  evidence with role masks and type priors.
- Retrieved subgraph export is implemented in `gems/visualization.py` and
  `scripts/export_gems_subgraph.py` with HTML, GraphML, JSON, and PNG outputs.

## Evaluation Check

- `scripts/evaluate_retrieval.py` compares ranked candidate endpoint IDs against
  the dataset positive endpoint ID from `selection_tasks.jsonl`.
- The metric is computed against dataset ground truth labels, not against
  another model's output.
- Default memory construction uses all endpoint catalog descriptions, but only
  train-split feedback events and train-split weak-supervision task traces as
  historical execution evidence.

## Non-Blocking Limitations

- The current semantic encoder is TF-IDF rather than a neural encoder.
- Reliability propagation is deterministic and untrained; it is suitable as a
  runnable baseline but not a learned GNN ablation.
- No live LLM service composer or real API executor is wired in yet; this code
  produces the graph memory and serialized evidence layer used by such a
  composer.
