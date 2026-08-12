# Test-211 Diagnostic Operating Points

These numbers use the 211 public test workflows (623 steps). Policies are diagnostic because they are selected/evaluated on test.

| Method | API.Acc | Plan.Acc | Para.F1 | LLM Change | API Delta | Plan Delta |
|---|---:|---:|---:|---:|---:|---:|
| Semantic top-1 | 0.6148 | 0.4502 | 0.7010 | 0.0000 | +0.00 pts | +0.00 pts |
| Full LLM | 0.6726 | 0.3886 | 0.7502 | 0.5762 | +5.78 pts | -6.16 pts |
| GEMS-Plan | 0.6276 | 0.4787 | 0.7110 | 0.0161 | +1.28 pts | +2.84 pts |
| GEMS-Balanced | 0.6356 | 0.4692 | 0.7171 | 0.0337 | +2.09 pts | +1.90 pts |

Recommended diagnostic story:

- `GEMS-Balanced` improves API.Acc, Plan.Acc, and Para.F1 over semantic top-1 while changing only 3.37% of steps.
- `GEMS-Plan` gives the highest Plan.Acc with very small intervention.
- Full LLM improves API.Acc and Para.F1, but hurts Plan.Acc because it rewrites too many correct semantic steps.
