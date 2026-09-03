# Marker-center multiradius geometry V23

V23 is a startable, zero-optimizer postprocessing candidate. It reuses the
V21 P1 ONNX payload and the fixed V13 synthetic `dev` split. Confidence `0.25`,
V13 proposal extraction and geometry filtering, offset decoding, radius
clipping, artifact-mask veto, and V21 radius-aware NMS remain unchanged.

The sole isolated change is marker geometry consensus. The existing
`support >= 3` ring rule is evaluated at every fixed integer radius from 3
through 12, and the candidate is accepted when any radius passes or the
existing center-density `>= 0.28` rule passes. No artifact-mask bypass is
allowed.

The authorized zero-optimizer P1 run produced 93 accepted candidates from 96 truth
centers: precision `1.0`, recall `0.96875`, F1 `0.9841269841269841`, zero
false positives, zero duplicates, zero prohibited-structure hits, and three
misses. It clears the 0.95 precision/recall bars and the zero prohibited-hit
bar. The tracked aggregate outcome is `P1_RESULT.json`. Production runtime
parity and held-out `real-sealed` acceptance remain required before approval.

Run the aggregate feasibility check from the repository root:

```powershell
python -m ml.markers.center.multiradius_geometry_v23.diagnose_v23 `
  --output ml/markers/center/multiradius_geometry_v23/V23_FEASIBILITY_DIAGNOSTIC.json
```

The runner in `candidate_runner.py` validates the exact V21 result, diagnostic,
and ONNX hashes, acquires the canonical candidate authorization, and records a
zero-optimizer aggregate dev result. Its separate `evaluate_candidate` helper
supports preparation tests without acquiring authorization. Neither path reads
public, sealed, private, article, or real data or emits case identities,
predictions, truth rows, or pixels.
