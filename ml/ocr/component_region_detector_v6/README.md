# Component-proposal graph text detector V6

This revision addresses a distinct OCR detection defect class after every
probability-map and DB-objective detector revision exhausted its fixed
three-candidate budget. The exposed official-detector diagnostic reached only
11 of 48 exact text scenes. A read-only follow-up showed that deterministic
connected-component line proposals preserve tight text geometry, while the
remaining problem is classifying those proposals without accepting markers,
axes, ticks, dividers, brackets, arrows, legends, connectors, or intersections.

V6 freezes new procedural numeric and generic-word scenes before training. Its
proposal algorithm, ordering, crop, eight geometry features, candidate IDs,
selection thresholds, CPU provider, parity tolerance, and zero-error scene
gates are fixed. The ignored public archive is checksum-bound and truth-hidden
from the future training runner. Chandler, `Generalization`, private and article
images, external datasets, pretrained weights, downloaded data, and exposed
predecessor fixture bytes are excluded.

P1 ran exactly once from its committed checksum-bound configuration and runner.
It passed all 48 validation scenes with 144/144 text regions, zero false
regions, misses, duplicates, or prohibited-structure hits at threshold `0.65`.
The exact CPU ONNX parity error was `2.86102294921875e-06`. P1 is consumed and
the only next authorized action is the once-only sealed public gate. The
candidate has not evaluated the sealed archive. No production manifest,
model-store entry, package
payload, production approval, release eligibility, or version 1.0.1 promotion
exists.
