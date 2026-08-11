# Marker-center background invariance V3

Runtime-consistency P2 used the exact radial marker payload and consistent
postprocessing. Its one public evaluation failed seven of twenty scenes with
three false positives and four misses. Five failed scenes used smooth
background-ramp degradation. Those exposed bytes are not reused here.

This defect class spends one zero-training candidate. It retains the exact
checkpoint, ONNX, proposal eligibility, text and artifact masks, confidence
threshold, geometry refinement, and duplicate suppression. Its only change is
subtracting the deterministic lower-middle median from the ink channel of each
33 by 33 proposal patch before ONNX execution. Fresh synthetic selection and
truth-hidden public families are frozen before inference. Selection and public
gates both require every scene exact, zero false positives, zero false
negatives, zero duplicates, zero prohibited-structure hits, and CPU parity no
worse than `1e-5`.

Passing this scientific gate cannot create a manifest or approve production.
Production still requires checksum-bound preprocessing parity in C#, an
approved mask provider, model-store discovery, packaging, private Chandler
validation, and clean-machine evidence.

## Consumed P1 result

P1 ran once from committed source with zero optimizer steps. CPU ONNX parity
passed at `1.430511474609375e-06`, but selection passed only 8 of 16 scenes.
It retained 125 true markers with five false positives, three false negatives,
three duplicates, and zero prohibited-structure hits. The public gate did not
execute. P1 is consumed and may not be rerun, tuned, manifested, stored,
packaged, or approved.
