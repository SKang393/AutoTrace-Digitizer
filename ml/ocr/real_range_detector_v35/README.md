# Corrected real-range learned proposal detector V35

V34's deterministic expansion did not resolve the raw proposal ceiling. The
committed V32 generator maximum-matched only 61 of 86 dev truths, recall
`0.7093023256`, so V32 and V33 classifier revisions could not reach the fixed
`0.95` proposal bars. V35 is therefore the first permitted learned proposal
generator after the approved pretrained, compatible fine-tune, project-owned
classifier, and deterministic expansion routes failed.

V35 trains a small project-owned source-scale segmentation model from V32's
five-axis family-disjoint corrected synthetic scenes. Truth boxes become tile
truth masks. Overlapping 256-pixel tiles with 64-pixel overlap are averaged in
original coordinates, thresholded at the fixed `0.40` probability point, and
converted to connected-component boxes. Tile offsets and edge clipping are
reversible and remain in original pixel coordinates. Raw proposals are scored
before any OCR classification using maximum-cardinality IoU `0.50` matching.

The authorized P1 run selected the lowest train-loss epoch and exported dynamic
CPU ONNX, but dev precision `0.20512820512820512` and recall
`0.46511627906976744` both failed. ONNX parity was
`1.0013580322265625e-05`, just above the `1e-5` limit. V35 is retired without a
private, article, public, or sealed-data read. The aggregate outcome is
`P1_RESULT.json`.

Verification:

```powershell
python -m pytest ml/ocr/real_range_detector_v35/tests -q
```
