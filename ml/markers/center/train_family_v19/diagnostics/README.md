# Marker-center V19 diagnostics

This directory records an aggregate-only diagnosis of the retired V19 P1
candidate on the fixed V13 synthetic dev split. It uses only the ignored V19
P1 ONNX payload under `artifacts/goal22-worktrees/marker-v19/.../P1-run` and
the project-owned procedural V13 dev generator. It does not read private or
article images, and it emits no scene IDs, case IDs, truth rows, pixels, or
per-case predictions.

Run from the repository root:

```powershell
python -m ml.markers.center.train_family_v19.diagnostics.diagnose_v19 `
  --output ml/markers/center/train_family_v19/diagnostics/V19_DIAGNOSTIC.json
```

The result reproduces V19 P1 at the fixed 0.25 threshold: 76 true positives,
zero false positives, and 20 false negatives. All 96 truths have a raw and
geometry-filtered proposal within 3 px. Nineteen misses have a best proposal
score below 0.25. One high-confidence miss is rejected by the existing
marker-geometry consensus veto. No proposal-availability or geometry-filter
recall defect is present in this split.

The next marker revision is startable only as a newly authorized candidate.
The recommended isolated change is classifier calibration or training coverage
for the low-confidence tail, while retaining an explicit guard for the
marker-geometry veto.
