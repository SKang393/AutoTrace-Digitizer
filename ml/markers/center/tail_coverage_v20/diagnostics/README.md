# V20 marker-center diagnostics

This directory contains the aggregate-only diagnosis for the retired V20 P1
candidate. `diagnose_v20.py` runs the fixed V13 synthetic `dev` split through
only the ignored V20 CPU ONNX payload and checks the ignored V20 checkpoint
binding. It reports proposal availability, classifier confidence, artifact and
marker-geometry vetoes, NMS/matching residuals, threshold sensitivity, family,
marker shape, and radius aggregates.

Run from the repository root:

```powershell
python -m ml.markers.center.tail_coverage_v20.diagnostics.diagnose_v20 --output ml/markers/center/tail_coverage_v20/diagnostics/V20_DIAGNOSTIC.json
python -m pytest ml/markers/center/tail_coverage_v20/diagnostics/test_diagnostics.py -q
```

The result is UTF-8 JSON with LF line endings. It never emits scene IDs,
participant or study names, pixels, truth rows, or per-case predictions. The
V20 P1 candidate remains synthetic-only, unconsumed, and not production or
release eligible. A subsequent revision is startable only with new candidate
authorization.
