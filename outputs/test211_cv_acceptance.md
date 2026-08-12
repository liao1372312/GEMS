# Test-211 Cross-Validated Acceptance

Diagnostic only: folds are created inside the 211-workflow test split.

| Method | API.Acc | Plan.Acc | Para.F1 | LLM Change |
|---|---:|---:|---:|---:|
| Semantic top-1 | 0.6148 | 0.4502 | 0.7010 | 0.0000 |
| Full LLM | 0.6726 | 0.3886 | 0.7502 | 0.5762 |
| CV-Accept | 0.7030 | 0.4171 | 0.7605 | 0.4767 |
| Oracle Sem/LLM | 0.9101 | 0.7867 | 0.9333 | 0.2953 |

Best CV threshold: `0.5`
