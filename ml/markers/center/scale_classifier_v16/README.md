# Marker-center scale classifier V16

V16 follows V15, which attempted fine-tuning
from the approved-compatible runtime-consistency P2 payload and failed the
fixed synthetic dev gate. Under the model-sourcing order, V16 therefore uses
the next permitted option: a new project-owned architecture trained from
scratch on project-owned synthetic data.

The model separates ink from text/artifact masks with two towers and retains
both 5x5 local and downsampled 3x3 context. It consumes the committed V13
geometry-filtered proposal stream and preserves the runtime contract
`[N,3,33,33] -> [N,4]`, including the 2.5 to 8 pixel radius head. ONNX export
checks candidate counts 1, 8, and 37 under CPU parity tolerance `1e-5`.

The authorized P1 dev loop completed two unconsumed attempts. Positive-loss
weight 4 reached precision `1.0` and recall `0.65625`; weight 16 reached
precision `1.0` and recall `0.6770833333333334`. Dynamic CPU ONNX parity
passed, but neither attempt cleared Tier 1. V16 is retired without a private,
public, or sealed-data read. The aggregate outcome is `P1_RESULT.json`.
