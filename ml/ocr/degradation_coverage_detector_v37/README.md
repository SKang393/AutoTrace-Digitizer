# Prepared OCR degradation-coverage detector V37

V35 and V36 both failed the fixed synthetic dev raw-proposal bars. V35 used
full-box pixel supervision and reached precision `0.20512820512820512` and
recall `0.46511627906976744`. V36 changed only the targets to centered shrink
cores and reached precision `0.15254237288135594` and recall
`0.313953488372093`. Its ground-truth expansion oracle and source tiling both
passed, isolating the remaining defect to learned pixel segmentation. V37
therefore rejects the V36 shrink-core target and restores the V35 full-box
target and postprocessing contract.

The only new factor is the synthetic train distribution. Each of the five V32
family-disjoint train scenes is retained byte-for-byte, then receives one fresh
deterministic composite variant applying blur, contrast, Gaussian noise, and a
JPEG round trip in sequence. The five composite variants span the five-point
aggregate grids without multiplying the training corpus fivefold; JPEG quality
spans the measured corrected-generator range `55` to `85`. Truth rectangles
and original pixel coordinates are unchanged. The V32 dev split is returned
directly and is never augmented, inspected for labels, or seeded from its
renderer families.

V37 retains the V35 detail-skip source-scale model, 256-pixel tiles with
64-pixel overlap, threshold `0.40`, minimum component area `8`, IoU `0.50`,
seed, AdamW settings, 12 epochs, CPU ONNX provider, dynamic parity batches,
Apache-2.0 license, and `0.95` raw-proposal bars. V35 and V36 result and
diagnostic hashes are bound exactly in the protocol and candidate config.

The authorized P1 run completed 312 optimizer steps over 408 train tiles and
selected epoch 11 from train loss. Fixed synthetic dev precision improves from
V35's `0.20512820512820512` to `0.23696682464454977`, and recall improves from
`0.46511627906976744` to `0.5813953488372093`, but both remain below the `0.95`
bars. CPU ONNX parity is `1.1444091796875e-05`, above the fixed `1e-5` limit.
V37 is retired without candidate consumption or a sealed, public, private,
article, or real-data read. The aggregate outcome is `P1_RESULT.json`.

Focused verification:

```powershell
python -m pytest ml/ocr/degradation_coverage_detector_v37/tests -q
```
