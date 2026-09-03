# V36 synthetic-dev diagnosis

`diagnose_v36.py` loads only the ignored V36 P1 ONNX checkpoint and the fixed
synthetic V32 `dev` split. It reports aggregate values for the four separable
stages:

- core-pixel threshold behavior against shrunken truth masks;
- connected components against shrunken truth boxes before expansion;
- deterministic expansion against source truth boxes;
- an oracle that expands ground-truth cores, isolating expansion geometry.

It also records source-scale tiling coverage and overlap disagreement plus
SHA-256 hashes for the checkpoint, fixed split source, and V36 protocol files.
No private or article data is opened, and no scene identifiers, pixels, truth
rows, or case-level predictions are written. The report is evidence for the
next OCR revision and does not authorize production approval or a sealed run.

Run from the repository root:

```text
python -m ml.ocr.shrink_region_detector_v36.diagnostics.diagnose_v36 \
  --checkpoint artifacts/goal22-worktrees/ocr-v36/ml/ocr/shrink_region_detector_v36/artifacts/P1-run/graph-text-shrink-region-detector-v36-p1.onnx \
  --output ml/ocr/shrink_region_detector_v36/diagnostics/DIAGNOSTIC.json
```
