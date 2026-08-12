# Processed Service Profile Dataset

This directory contains processed data for confidence-aware dynamic service
profile learning from the raw RapidAPI-style JSON files in `dataset/api`.

## Files

- `services.jsonl`: one record per service. Includes category, description,
  host, pricing, QoS summary, initial confidence, profile text, and split.
- `endpoints.jsonl`: one record per service endpoint. Includes method, URL,
  endpoint description, compact parameter lists, schema summary, and split.
- `feedback_events.jsonl`: one compact execution-feedback event per endpoint
  with available `test_endpoint` data. Large raw responses are replaced by
  shape, top-level keys, preview text, status, and confidence signals.
- `selection_tasks.jsonl`: weakly supervised service/endpoint selection tasks.
  Each task has one positive endpoint and six negative candidate endpoints from
  the same split, with hard negatives sampled from the same category when
  possible.
- `splits/service_splits.json`: deterministic service-level train/val/test
  split map.
- `stats.json`: build statistics and output paths.

## Split Policy

Splits are assigned at the service level with seed `42`:

- train: 2759 services, 12537 endpoints
- val: 345 services, 1648 endpoints
- test: 345 services, 1720 endpoints

No endpoint, feedback event, or selection-task candidate crosses service split
boundaries.

## Intended Experiment Use

- Static service understanding: use `services.jsonl` and `endpoints.jsonl`.
- Confidence-aware service ranking: use QoS fields and `initial_confidence`.
- Dynamic feedback calibration: use `feedback_events.jsonl`.
- Tool/service selection evaluation: use `selection_tasks.jsonl`.

Regenerate with:

```bash
python scripts/build_service_profile_dataset.py --input dataset/api --output dataset/processed
```
