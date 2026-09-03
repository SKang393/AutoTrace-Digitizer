# Marker-center mask-preserving V24 feasibility

This is a zero-optimizer, non-consuming feasibility artifact. It reuses the
exact V21 P1 ONNX payload at confidence threshold `0.25`, with CPU
`CPUExecutionProvider`, unchanged `[N,3,33,33] -> [N,4]` tensors, offset/radius
decoding, V23 multiradius consensus, radius-aware NMS, and prohibited-structure
checks.

The isolated change is proposal extraction: every ink-supported proposal is
emitted even when OCR or artifact masks cross its center. Both masks remain in
channels 1 and 2. Postprocessing removes only the hard center-mask rejection.
The corrected `real_range_generator_v1` synthetic `dev` split is the only data
read. No candidate budget, optimizer, private corpus, real-sealed split, model
artifact, manifest, or promotion is created.

The frozen V21 payload retains complete proposal coverage but fails the fixed
dev gate at precision `0.6936936936936937`, recall `0.499500998003992`, and 167
prohibited hits. The zero-optimizer feasibility attempt consumes no candidate;
one preregistered training candidate is therefore startable.

Run from the repository root when the explicitly bound V21 model is available:

```powershell
python -m ml.markers.center.mask_preserving_v24.diagnose_v24 --model artifacts/goal22-worktrees/marker-v21/ml/markers/center/focal_confidence_v21/artifacts/P1-run/marker-center-focal-confidence-v21-p1.onnx --output ml/markers/center/mask_preserving_v24/FEASIBILITY.json
```

Focused tests:

```powershell
python -m pytest ml/markers/center/mask_preserving_v24/tests -q
```
