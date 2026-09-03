# Prepared DB-style shrink-region detector V36

V35 was retired after source-scale full-box supervision and connected-component
recovery reached only 0.205 precision and 0.465 recall on its fixed synthetic
dev split. The checksum-bound V35 diagnosis shows that the pixel segmentation
ceiling, not threshold or morphology selection, was responsible.

V36 is the smallest preregistered repair. It keeps the V35 detail-skip model,
source-scale 256-pixel tiles with 64-pixel overlap, fixed family-disjoint V32
train/dev scenes, seed, AdamW settings, 12 epochs, CPU ONNX provider, parity
batches, license, and 0.95 raw-proposal bars. Each truth rectangle is instead
represented by one DB-style centered rectangular core using a fixed 0.40 ratio
on each axis. Core components are thresholded without morphology, then expanded
once with the same fixed axis ratio and source-canvas clipping. The recovered
full extent uses deterministic half-up integer rounding before its added pixels
are split around the core center.

`ShrinkGeometry` retains integer side insets so audit reversal is exact for
ground-truth geometry. Predicted cores use the same centered axis ratio and
deterministic clipping for source-box recovery. Separate cores are never merged
by a closing or dilation step, which preserves adjacent text-region identity
before expansion.

The authorized P1 run completed 156 optimizer steps and selected epoch 12 from
train loss. Fixed synthetic dev precision is `0.15254237288135594` and recall
is `0.313953488372093`, both worse than V35 and below the `0.95` bars. Dynamic
CPU ONNX parity passes at `9.5367431640625e-06`. V36 is retired without a
sealed, public, private, article, or real-data read and without candidate
consumption. The tracked aggregate outcome is `P1_RESULT.json`.

Focused verification:

```powershell
python -m pytest ml/ocr/shrink_region_detector_v36/tests -q
```
