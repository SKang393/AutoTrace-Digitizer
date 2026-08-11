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

The split-freeze checkpoint itself authorized no optimizer execution. P1 now
has a separately checksum-bound small export-safe CNN runner and configuration;
the canonical budget authorizes that exact committed pair only. Nothing here
creates a production manifest, enters the model store, packages weights,
changes release readiness, or approves version 1.0.1.
