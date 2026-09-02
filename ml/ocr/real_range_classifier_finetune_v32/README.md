# Corrected real-range classifier fine-tune V32

This revision adapts the existing checksum-bound V10 P2 proposal
classifier to the corrected synthetic real-range proposal distribution. It
changes classifier weights only. Proposal generation, the 0.82 operating
threshold, and canonical maximum-cardinality IoU 0.50 matching remain fixed.

The train and dev scenes use disjoint renderer, font, degradation, template,
and marker families while covering the corrected source-size range. The runner
uses the canonical training-budget authorization, performs dynamic CPU ONNX
parity, and evaluates dev aggregates only. It never opens public or sealed
fixtures.

Two unconsumed dev attempts completed. The baseline fine-tune reached precision
`0.855072463768116` and recall `0.686046511627907`. A fixed positive-class
weight of 4 reached precision `0.8333333333333334` and recall
`0.6976744186046512`. Both remain far below Tier 1, so V32 is retired without a
sealed read or candidate consumption. The aggregate outcome is
`P1_RESULT.json`.

Verification:

```powershell
python -m pytest ml/ocr/real_range_classifier_finetune_v32/tests -q
python -m pytest ml/ocr/real_range_classifier_finetune_v32/tests -q
```
