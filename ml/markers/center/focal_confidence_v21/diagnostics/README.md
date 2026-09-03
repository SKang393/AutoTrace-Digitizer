# Marker-center focal confidence V21 diagnostics

This diagnostic runs the ignored V21 P1 ONNX payload against only the fixed
V13 synthetic `dev` split. It reports aggregate proposal availability,
confidence, artifact and marker-geometry vetoes, NMS or matching residuals,
family, marker shape, marker radius, threshold sensitivity, and artifact
hashes. It never emits scene IDs, case details, pixels, predictions, or truth
rows.

The V21 payloads are not copied into the repository. Run from the repository
root with the authorized ignored artifacts:

```powershell
python -m ml.markers.center.focal_confidence_v21.diagnostics.diagnose_v21 `
  --output ml/markers/center/focal_confidence_v21/diagnostics/V21_DIAGNOSTIC.json
```

The result is UTF-8 with LF line endings. A new marker revision is startable
only with new candidate authorization. The fixed-dev result attributes seven
of eight misses to downstream artifact-mask or marker-geometry vetoes and one
to confidence, with no proposal-availability or NMS defect. This diagnostic
itself does not train, open a sealed/public split, or change any model.
