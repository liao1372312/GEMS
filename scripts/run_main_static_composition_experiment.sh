#!/usr/bin/env bash
set -euo pipefail

# Main static-memory composition experiment helper.
#
# By default this runs only local baselines and table aggregation. To run the
# external LLM-agent prompt baselines, set RUN_LLM_BASELINES=1 with LLM_API_KEY
# or OPENAI_API_KEY configured.

python scripts/run_public_composition_experiments.py \
  --output outputs/public_composition_experiments.json

python scripts/evaluate_reflexion_memory_baseline.py \
  --output outputs/reflexion_memory_baseline.json

if [[ "${RUN_LLM_BASELINES:-0}" == "1" ]]; then
  python scripts/llm_public_composition_baselines.py \
    --split test \
    --limit 0 \
    --max-steps 0 \
    --baselines direct_llm,cot_llm,react,restgpt,ma_nomem \
    --output outputs/llm_public_composition_baselines_test_all.json
fi

python scripts/export_main_experiment_table.py
