# OCR detector V38: equal-weight batch Dice preparation

V38 is a single prepared candidate for Goal 22 Phase 4R. It retains V37's
ten synthetic training scenes, V35's source-scale model, full-box targets,
tiling, and postprocessing, and V32's fixed five-scene dev split.

The isolated change is the pixel objective. The V37 positive-weighted
BCE-with-logits term is unchanged and is summed with one batch soft-Dice loss.
The Dice term uses fixed `epsilon = 1e-6` and reduces over the complete batch.
No threshold, model, data, optimizer, provider, parity, or acceptance bar was
changed.

The authorized P1 run completed 312 optimizer steps over 408 train tiles and
selected epoch 11 from train loss. Fixed synthetic dev precision is
`0.22169811320754718` and recall is `0.5465116279069767`, both slightly below
V37 and far below the `0.95` bars. CPU ONNX parity is
`1.33514404296875e-05`, above the fixed `1e-5` limit. V38 is retired without
candidate consumption or a sealed, public, private, article, or real-data read.
The aggregate outcome is `P1_RESULT.json`.
