# Marker classifier acceptance preregistration

This preregistration was written before the held-out `test` split was opened.
The Session 11 instruction requires an agreed held-out gate but does not supply
a maintainer-approved number. This session therefore uses the following local
engineering gates without representing them as maintainer-agreed:

- shape macro-F1 across the nine non-artifact shape classes: at least `0.90`;
- fill macro-F1 across `filled`, `open`, and `unknown`: at least `0.90`;
- ONNX CPU parity maximum absolute error: at most `1e-5`.

Model and temperature selection use only the fixed training and validation
families. The held-out split is opened once by `benchmark.py`, after checkpoint
and ONNX selection. The benchmark creates an exclusive seal and refuses a
second run. A failed result remains failed and must not trigger test-set tuning.

The experiment budget is three. The initial implementation preselects one
compact architecture and does not spend the two remaining slots unless the
validation result, not the held-out result, requires a documented comparison.
