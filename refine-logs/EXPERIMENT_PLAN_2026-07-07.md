# Experiment Plan

**Problem**: Adaptive service composition with reusable historical execution memory.
**Method Thesis**: GEMS improves static-memory service composition by retrieving execution-grounded graph evidence and selectively accepting helpful LLM/memory corrections.
**Date**: 2026-07-07

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: Static-memory GEMS improves composition quality | This is the main paper claim: given the same request, service catalog, and training-trace memory, GEMS should generate better workflows than LLM/RAG/GraphRAG baselines | Held-out test improvement on API.Acc and Para.F1, with workflow exact-match trade-off reported honestly | B1 |
| C2: Reliability evidence should be selective | Reliability helps under hazards but can hurt clean semantic matching if applied everywhere | Endpoint gate and hazard stress results showing clean retrieval preservation plus hazard-safe gains | B2, B3 |

## Paper Storyline

- Main paper must prove: GEMS is useful in a static-memory end-to-end composition setting.
- Main paper can also support: reliability and graph memory are useful only when selectively activated.
- Appendix / auxiliary analysis can support: temporal proxy behavior and diagnostic routing details.
- Experiments intentionally not claimed as main evidence: live online repair, true runtime execution success, and full outcome-driven update under real service evolution.

## Experiment Blocks

### Block 1: Static-Memory End-to-End Composition

- Claim tested: C1.
- Why this block exists: This is the main experiment. It asks whether GEMS improves service composition when all methods use the same current request, current service catalog, and fixed memory built from training-stage traces.
- Dataset / split / task: Public composition benchmark from `five_domain_1400_gold.jsonl`; train 979 requests / 2859 steps, val 210 / 604, test 211 / 623.
- Compared systems: Direct-LLM, CoT-LLM, ReAct, RestGPT-style, MA-NoMem, Trace-RAG, StructMem-RAG, GraphRAG-static, Reflexion-Memory, GEMS.
- Metrics: API.Acc, Plan.Acc, Para.F1; API.Top-3 and API.MRR for ranking-only methods.
- Setup details: Train split builds memory; validation split selects verifier policies; test split is held out. No test-time memory update is used in the main result.
- Success criterion: GEMS should improve API.Acc and Para.F1 over completed LLM, agent, RAG, GraphRAG, and experience-memory baselines while reporting any Plan.Acc trade-off.
- Failure interpretation: If Plan.Acc remains below semantic top-1, the claim should be selective correction rather than full workflow dominance.
- Table / figure target: Main paper Table `tab:main-results`.
- Priority: MUST-RUN.

### Block 2: Endpoint Retrieval Gate

- Claim tested: C2.
- Why this block exists: It isolates provider-role retrieval and checks whether reliability should be unconditional or gated.
- Dataset / split / task: Processed endpoint retrieval tasks under `dataset/processed`; 1720 provider-role test tasks.
- Compared systems: Text-only, no reliability, text + reliability, GEMS full retrieval, GEMS-Selective, reliability-only, initial-confidence-only.
- Metrics: Top-1, Top-3, MRR.
- Setup details: GEMS-Selective preserves semantic/no-reliability ranking unless a strong risk or conflict signal activates reliability-aware scoring.
- Success criterion: GEMS-Selective should match the best clean retrieval while full reliability ablations show why unconditional reliability is unsafe.
- Failure interpretation: If GEMS-Selective is below text-only, the gate is too aggressive.
- Table / figure target: Main paper Table `tab:retrieval-ablation-results`.
- Priority: MUST-RUN.

### Block 3: Reliability Stress and Hazard Robustness

- Claim tested: C2.
- Why this block exists: It verifies that reliability evidence matters under stale, failed, or conflicting memories.
- Dataset / split / task: Reliability stress groups and public hazard proxy slices.
- Compared systems: Text-only, no reliability, text + reliability, GEMS full retrieval, reliability-only, semantic, Trace-RAG, GraphRAG.
- Metrics: Success@1, Success@3, MRR-first-success, safe top-ranked endpoint rate, hazard-slice API.Acc.
- Setup details: Stress cases intentionally make failed endpoints semantically attractive.
- Success criterion: Reliability-aware variants should improve safety under hazards, even if reliability-only is not suitable for ordinary matching.
- Failure interpretation: If reliability does not help stress cases, the execution-grounded memory signal is not useful.
- Table / figure target: Main paper reliability stress and hazard tables.
- Priority: MUST-RUN.

### Block 4: Industrial EMP External Benchmark

- Claim tested: Dataset dependence and external validity.
- Why this block exists: It checks whether the result generalizes to a larger industrial-style benchmark.
- Dataset / split / task: EMP synthetic composition benchmark; train 700, val 150, test 150.
- Compared systems: Semantic top-1, Trace-RAG, GEMS w/o quality, GEMS + quality.
- Metrics: API.Acc, API.Top-3, API.MRR, Plan.Acc, Sim.Exec.SR, Para.F1.
- Setup details: Sim.Exec.SR is simulated from catalog quality metadata and workflow correctness.
- Success criterion: Identify whether historical traces or quality-aware graph signals help on a different benchmark.
- Failure interpretation: If Trace-RAG is strongest, report this as dataset dependence rather than overclaiming GEMS dominance.
- Table / figure target: EMP result table.
- Priority: MUST-RUN.

### Block 5: Verifier, Router, and Cost Ablations

- Claim tested: C1 mechanism and efficiency.
- Why this block exists: It explains why selective verification is needed instead of all-step LLM reranking.
- Dataset / split / task: Public composition validation/test.
- Compared systems: All-step LLM, confidence router, validation-trained acceptance, GEMS-Plan, GEMS-API, GEMS-Adaptive.
- Metrics: API.Acc, Plan.Acc, Para.F1, LLM call rate, change rate, token cost.
- Setup details: Policies are selected on validation and evaluated once on test.
- Success criterion: GEMS-API should provide the strongest API/Para result among selective policies, with Plan.Acc trade-offs disclosed.
- Failure interpretation: If router/verifier does not beat all-step LLM, selective acceptance is not adding value.
- Table / figure target: Routing/verifier ablation table.
- Priority: MUST-RUN.

### Block 6: Temporal Proxy Analysis

- Claim tested: Not a main claim; boundary check for online update.
- Why this block exists: The method includes outcome-driven update, but current data does not support a live online adaptation claim.
- Dataset / split / task: Deterministic public test batches.
- Compared systems: Semantic, Trace-RAG, GraphRAG, GEMS heuristic.
- Metrics: Workflow exact by batch.
- Setup details: Treated as auxiliary only; not used to claim online adaptation.
- Success criterion: The analysis should clarify limitations and prevent overclaiming.
- Failure interpretation: Negative or non-monotonic results mean live online update must be future work or a separate experiment.
- Table / figure target: Auxiliary temporal proxy table.
- Priority: NICE-TO-HAVE.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | Verify data and metrics | dataset stats, semantic baseline | Metrics match exported tables | low | metric mismatch |
| M1 | Main static-memory baseline | Direct-LLM, CoT-LLM, ReAct, RestGPT-style, MA-NoMem, Trace-RAG, StructMem, GraphRAG, Reflexion-Memory | Baselines stable on test | medium/high due LLM | missing full LLM baseline runs |
| M2 | Main GEMS result | GEMS-Plan/API/Adaptive | GEMS improves API/Para on held-out test | medium/high due LLM | Plan.Acc trade-off |
| M3 | Mechanism isolation | endpoint gate, hazard stress | Selective reliability story holds | low | reliability hurts clean retrieval |
| M4 | External validity | EMP benchmark | Report positive or negative honestly | low | Trace-RAG may dominate |
| M5 | Polish | routing/cost/error analysis | Tables support narrative | low | too many diagnostics in main text |

## Final Checklist

- [x] Main paper has a static-memory end-to-end composition setting.
- [x] Main result does not rely on online memory update.
- [x] Validation selects policies; test reports final numbers.
- [x] Reliability is framed as selective, not universal.
- [x] Temporal proxy is separated from the main claim.
