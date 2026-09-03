# V38 false-positive attribution

This diagnostic holds the ignored V38 CPU ONNX payload fixed and evaluates the
fixed V32 synthetic `dev` split at threshold `0.40`. It emits aggregate pixel
and connected-component false-positive counts, plus aggregate false-negative
counts by truth-box size and source dimension.

The generator provides a marker mask but does not provide raster masks for all
structures. The report therefore uses deterministic geometric proxies:

- axes and ticks: annotation lines with 7-pixel thickness;
- marker and connecting-line ink: the supplied marker mask, marker radius plus
  2 pixels, and annotated edge lines with 3-pixel thickness;
- phase dividers: annotated divider lines with 5-pixel thickness;
- text-box margins: annotated rendered text boxes expanded by 3 pixels;
- remaining pixels: source grayscale below 200 are `other_dark_ink`, and the
  rest are `empty_background`.

Proxy masks are disjoint in that order. Component attribution chooses the
category with the most pixels in each unmatched component, breaking ties by the
same order. No scene identifiers, pixels, truth rows, or per-case predictions
are written to the JSON report.

Run from the repository root:

```powershell
python -m ml.ocr.dice_loss_detector_v38.diagnostics.fp_attribution.attribute `
  --output ml/ocr/dice_loss_detector_v38/diagnostics/fp_attribution/ATTRIBUTION.json
python -m pytest ml/ocr/dice_loss_detector_v38/diagnostics/fp_attribution/tests -q
```

The ignored ONNX payload must be present at the V38 Goal 22 worktree path and
must match the checksum bound in `attribute.py`.
