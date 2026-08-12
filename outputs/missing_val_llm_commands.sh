#!/usr/bin/env bash
set -euo pipefail

# Run this in a network-enabled environment with LLM_API_KEY/OPENAI_API_KEY configured.
python scripts/llm_public_composition_rerank.py \
  --split val \
  --limit 0 \
  --case-filter all \
  --continue-on-error \
  --output outputs/llm_public_composition_rerank_val_all.json

python scripts/evaluate_llm_acceptance_router.py \
  --llm-files outputs/llm_public_composition_rerank_val_all.json outputs/llm_public_composition_rerank_test_all.json \
  --output outputs/llm_acceptance_router_eval.json

python scripts/evaluate_gems_plan_verifier.py \
  --llm-files outputs/llm_public_composition_rerank_val_all.json outputs/llm_public_composition_rerank_test_all.json \
  --output outputs/gems_plan_verifier_eval.json

python scripts/evaluate_gems_plan_verifier.py \
  --objective api \
  --llm-files outputs/llm_public_composition_rerank_val_all.json outputs/llm_public_composition_rerank_test_all.json \
  --output outputs/gems_plan_verifier_api_objective_eval.json

python scripts/export_experiment_tables.py

python scripts/audit_paper_readiness.py
