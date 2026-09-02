# Marker-center geometry fine-tune V15

V15 fine-tunes the technically compatible radial-topology proposal classifier
from the checksum-bound runtime-consistency P2 checkpoint on the committed V13
geometry-filtered synthetic train stream, then evaluates only the fixed
synthetic dev stream.

This follows model sourcing order because an approved existing payload is
reused before any new weights are trained. The runtime radius contract remains
2.5 to 8 pixels, and ONNX export uses a dynamic candidate-count axis with a
`1e-5` parity limit.

The authorized P1 run completed 342 optimizer steps and passed dynamic CPU ONNX
parity, but failed dev at every fixed threshold. Its best F1 operating point,
threshold `0.70`, reached precision `0.3959731543624161` and recall
`0.6145833333333334`. The candidate was retired without reading a sealed or
public split and without private data. The aggregate outcome is
`P1_RESULT.json`.
