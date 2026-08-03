# Validation-only training comparison

The held-out test split was sealed by content hash before training and was not
evaluated during these repairs. Exactly three threshold values are declared in
the final model-selection run: 0.32, 0.36, and 0.40.

Only input-family and degradation-family identifiers are disjoint. Target
geometries and layout templates repeat across splits, so the comparison does
not demonstrate generalization to new geometries, layouts, or real articles.

| Attempt | Change | Standard validation F1 at 5 px | Zero-mask F1 at 5 px | Exact scenes | Decision |
|---|---|---:|---:|---:|---|
| 1 | Initial focal/radius/artifact training | 0.0000 | 0.0000 | 0/3 | Artifact loss mislabeled marker ink as artifact. |
| 2 | Balanced artifact loss plus marker-negative penalty | 0.8387 | 0.8387 | 0/3 | Spatial peaks were correct but confidence was under-calibrated. |
| 3 | Fixed +2.0 center-logit calibration after training | 1.0000 | 1.0000 | 3/3 | Selected threshold 0.36; proceed to one held-out evaluation. |

All three attempts used the same architecture, training families, seed,
90-epoch schedule, and mask-dropout modes. Attempt 3 changes confidence
calibration only and does not change peak locations.

The single sealed held-out evaluation was then run once. Standard-mask results
were F1 0.9870 at both 3 px and 5 px, 38/38 true centers, one unrelated false
positive, zero duplicates, and zero hits for all eight hard-negative kinds.
Zero-mask ablation was F1 0.9620 with three false positives, including two
legend hits. No model or threshold change was made after opening held-out data.
