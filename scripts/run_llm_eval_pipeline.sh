#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/llm_public_composition_rerank.py \
  --split test \
  --case-filter non_top1 \
  --limit 100 \
  --output outputs/llm_public_composition_rerank_non_top1_100.json

python scripts/summarize_llm_rerank_results.py \
  --output outputs/llm_rerank_summary.json

python scripts/export_experiment_tables.py \
  --output outputs/experiment_tables.tex

echo "LLM evaluation pipeline complete."
echo "Summary: outputs/llm_rerank_summary.json"
echo "Tables: outputs/experiment_tables.tex"
