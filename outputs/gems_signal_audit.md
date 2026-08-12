# GEMS Signal Audit

## Val

- Steps: 604
- Candidate rows: 6040
- Gold reliability mean: 0.8838
- Non-gold reliability mean: 0.8666

| Feature | ROC-AUC | AP | Pos Mean | Neg Mean | Top-1 | Workflow Exact |
|---|---:|---:|---:|---:|---:|---:|
| semantic_similarity | 0.6373 | 0.1948 | 0.2993 | 0.2771 | 0.6507 | 0.4571 |
| semantic_rank_reciprocal | 0.8473 | 0.4990 | 0.7448 | 0.2427 | 0.6507 | 0.4571 |
| trace_score | 0.6334 | 0.2434 | 0.1310 | 0.0270 | 0.4851 | 0.2238 |
| graph_score | 0.6746 | 0.2010 | 0.4200 | 0.3802 | 0.3791 | 0.1333 |
| node_reliability | 0.5834 | 0.1389 | 0.8838 | 0.8666 | 0.1871 | 0.0571 |
| node_risk | 0.5050 | 0.0997 | 0.0172 | 0.0152 | 0.4503 | 0.2238 |

## Test

- Steps: 623
- Candidate rows: 6230
- Gold reliability mean: 0.8741
- Non-gold reliability mean: 0.8622

| Feature | ROC-AUC | AP | Pos Mean | Neg Mean | Top-1 | Workflow Exact |
|---|---:|---:|---:|---:|---:|---:|
| semantic_similarity | 0.6298 | 0.1779 | 0.2945 | 0.2751 | 0.6148 | 0.4502 |
| semantic_rank_reciprocal | 0.8204 | 0.4552 | 0.7139 | 0.2461 | 0.6148 | 0.4502 |
| trace_score | 0.6110 | 0.1714 | 0.0976 | 0.0268 | 0.3933 | 0.1517 |
| graph_score | 0.6610 | 0.1719 | 0.4207 | 0.3874 | 0.3194 | 0.1327 |
| node_reliability | 0.5504 | 0.1373 | 0.8741 | 0.8622 | 0.1830 | 0.0427 |
| node_risk | 0.4990 | 0.1005 | 0.0160 | 0.0172 | 0.4125 | 0.2133 |

## Interpretation

- If graph/reliability AUC is near 0.5 or lower than semantic similarity, the graph memory is not providing a useful gold-selection signal for this benchmark.
- If reliability is similar for gold and non-gold candidates, reliability propagation cannot improve API selection without better execution labels.
- Strong semantic-rank signal means the dataset is still dominated by the candidate generator's original ranking.
