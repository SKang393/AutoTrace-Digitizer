# Balanced-recall graph text detector V2

This is a distinct OCR-detection defect class created after
`graph-text-region-detector-v1` exhausted P1 through P3. The prior final model
passed every graph-structure exclusion but produced no matched region on 29 of
72 text fixtures and left ten false regions. Provider execution and the strict
probability contract passed, so the new responsible subsystem is sparse-positive
feature preservation and recall.

P1 replaces the exhausted tiny strided model and unweighted loss with a
skip-connected encoder-decoder, a fixed positive BCE weight of `8.0`, and a
fixed Dice weight of `2.0`. It retains BGR normalization, the DB shrink ratio
`0.40`, probability threshold `0.30`, box threshold `0.60`, unclip ratio `1.5`,
minimum side `3`, and maximum regions `1000`.

Training, selection, and sealed-public data use new procedural renderer and
degradation families. They do not reuse V1 selection or sealed fixtures and do
not contain Chandler, Generalization, private images, article images, external
datasets, pretrained weights, or downloaded training data.

Selection requires all 96 fixtures exact, zero false regions, zero duplicates,
zero exclusion regions, CPU execution, and ONNX parity at most `1e-4`. P1 is
the only authorized candidate. The sealed public archive remains unopened.
No result in this folder can create a production manifest, populate the model
store, package weights, approve the combined OCR pair, change release readiness,
or authorize version `1.0.1` by itself.
