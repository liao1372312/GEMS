# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M1 | Static-memory baseline | Semantic / Trace-RAG / StructMem / GraphRAG | test | API.Acc, Plan.Acc, Para.F1 | MUST | DONE | Main baseline table populated |
| R002 | M2 | All-step LLM comparison | LLM reranker | val/test | API.Acc, Plan.Acc, Para.F1, tokens | MUST | DONE | Full val/test coverage |
| R003 | M2 | GEMS verifier policies | GEMS-Plan / GEMS-API / GEMS-Adaptive | val/test | API.Acc, Plan.Acc, Para.F1, change rate | MUST | DONE | Policies selected on validation |
| R004 | M1 | No-memory LLM baselines | Direct-LLM / CoT-LLM | test | API.Acc, Plan.Acc, Para.F1 | MUST | TODO | Partial 100-step run exists; needs full test |
| R005 | M1 | Tool and REST agent baselines | ReAct / RestGPT-style | test | API.Acc, Plan.Acc, Para.F1 | MUST | TODO | Prompt script exists; needs full test |
| R006 | M1 | Multi-agent no-memory baseline | MA-NoMem | test | API.Acc, Plan.Acc, Para.F1 | MUST | TODO | Prompt script exists; needs full test |
| R007 | M1 | Experience-memory baseline | Reflexion-Memory | test | API.Acc, Plan.Acc, Para.F1 | SHOULD | DONE | Implemented textual reflection memory baseline; test API.Acc 0.4446, Plan.Acc 0.1991, Para.F1 0.5657 |
| R008 | M3 | Endpoint retrieval gate | GEMS-Selective and ablations | test | Top-1, Top-3, MRR | MUST | DONE | GEMS-Selective tied best |
| R009 | M3 | Reliability stress | reliability variants | test | Success@1, MRR-first-success | MUST | DONE | Supports selective reliability |
| R010 | M4 | External benchmark | EMP methods | test | API.Acc, Plan.Acc, Sim.Exec.SR, Para.F1 | MUST | DONE | Trace-RAG strongest; reported as dataset dependence |
| R011 | M5 | Routing/cost ablation | routers and verifiers | test | API.Acc, Plan.Acc, Para.F1, cost | MUST | DONE | Supports selective verification |
| R012 | M5 | Temporal boundary check | batch proxy | test batches | Workflow Exact | NICE | DONE | Auxiliary only, not main claim |
