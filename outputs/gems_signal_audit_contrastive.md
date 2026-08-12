# GEMS Signal Audit

## Val

- Steps: 604
- Candidate rows: 6040
- Gold reliability mean: 0.8291
- Non-gold reliability mean: 0.7901

| Feature | ROC-AUC | AP | Pos Mean | Neg Mean | Top-1 | Workflow Exact |
|---|---:|---:|---:|---:|---:|---:|
| semantic_similarity | 0.6373 | 0.1948 | 0.2993 | 0.2771 | 0.6507 | 0.4571 |
| semantic_rank_reciprocal | 0.8473 | 0.4990 | 0.7448 | 0.2427 | 0.6507 | 0.4571 |
| trace_score | 0.6334 | 0.2434 | 0.1310 | 0.0270 | 0.4851 | 0.2238 |
| graph_score | 0.6481 | 0.1976 | 0.3843 | 0.3382 | 0.3328 | 0.1190 |
| node_reliability | 0.6355 | 0.1633 | 0.8291 | 0.7901 | 0.2053 | 0.0571 |
| node_risk | 0.4829 | 0.1083 | 0.1652 | 0.1571 | 0.1589 | 0.0524 |

## Test

- Steps: 623
- Candidate rows: 6230
- Gold reliability mean: 0.8178
- Non-gold reliability mean: 0.7907

| Feature | ROC-AUC | AP | Pos Mean | Neg Mean | Top-1 | Workflow Exact |
|---|---:|---:|---:|---:|---:|---:|
| semantic_similarity | 0.6298 | 0.1779 | 0.2945 | 0.2751 | 0.6148 | 0.4502 |
| semantic_rank_reciprocal | 0.8204 | 0.4552 | 0.7139 | 0.2461 | 0.6148 | 0.4502 |
| trace_score | 0.6110 | 0.1714 | 0.0976 | 0.0268 | 0.3933 | 0.1517 |
| graph_score | 0.6336 | 0.1638 | 0.3821 | 0.3464 | 0.2809 | 0.1232 |
| node_reliability | 0.5980 | 0.1597 | 0.8178 | 0.7907 | 0.1862 | 0.0427 |
| node_risk | 0.4892 | 0.1098 | 0.1574 | 0.1474 | 0.1445 | 0.0521 |

## Interpretation

- If graph/reliability AUC is near 0.5 or lower than semantic similarity, the graph memory is not providing a useful gold-selection signal for this benchmark.
- If reliability is similar for gold and non-gold candidates, reliability propagation cannot improve API selection without better execution labels.
- Strong semantic-rank signal means the dataset is still dominated by the candidate generator's original ranking.
