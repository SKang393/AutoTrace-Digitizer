# Marker-center V22 veto-override feasibility

This is an aggregate-only, zero-optimizer feasibility check for one bounded
runtime postprocessing change. It runs the ignored V21 P1 ONNX payload through
the fixed synthetic V13 `dev` split and retains V21's proposal extractor,
offset decoding, radius clipping, confidence threshold `0.25`, ordinary
artifact-mask and marker-geometry vetoes, and radius-aware NMS.

The only swept parameter is a generic high-confidence floor. A candidate that
passes the retained `0.25` threshold but is rejected by an artifact-mask or
marker-geometry veto may bypass that veto only when its confidence reaches the
floor. No public, sealed, private, article, or real corpus is read, and no
training or optimizer step is performed.

Run from the repository root with the authorized ignored V21 artifacts:

```powershell
python -m ml.markers.center.veto_override_v22.diagnose_v22 `
  --output ml/markers/center/veto_override_v22/V22_FEASIBILITY_DIAGNOSTIC.json
```

The fixed V13 manifest, V21 result, V21 diagnostic, and ONNX payload hashes
are recorded in the aggregate diagnostic. It emits no scene
IDs, case detail, predictions, truth rows, or pixels.

The 2026-09-02 sweep failed feasibility: floor `0.90` admitted one vetoed
candidate and reached recall `0.9270833333333334` at precision `1.0`; floors
`0.95`, `0.99`, `0.995`, and `0.999` admitted none and retained recall
`0.9166666666666666`. All floors had zero prohibited hits, but none cleared
both `0.95` precision and recall bars. Therefore no V22 candidate protocol,
config, or training runner is created.
