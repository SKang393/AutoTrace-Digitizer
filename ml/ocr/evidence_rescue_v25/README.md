<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# OCR V25 evidence-rescue candidate

V25 is a fresh project-owned OCR proposal and role defect class. Its design
uses only aggregate terminal evidence from consumed V23 P3 and V24 P2. No
predecessor case identity, truth record, fixture pixel, private image,
Chandler image, or `Generalization` label informed the design.

V23 P3 retained all 1,024 visible-selection truths but also retained three
false prohibited regions. V24 P2 removed every false region but missed two
truths while preserving the V23 role outputs. V25 P1 therefore keeps the exact
frozen V24 P2 model, preserves its roles, retains every V24 P2 acceptance, and
rescues only a parent-accepted rejection whose already-bound CTC evidence
passes fixed generic confidence, entropy, blank, length, and alphanumeric
conditions.

P1 performs zero optimizer steps. The complete split was frozen once from
source commit `9805c2db397e2b7857093b3292cf115ddb6d559b`. The train,
validation, and truth-hidden public archives contain 256, 128, and 192 fresh
scenes and have SHA-256 values `63f9d63a...bcc1`, `20284a9a...f9d`, and
`bab5eaa6...c40`. Their source-byte overlap is zero. The train split remains
available only for a later separately preregistered candidate if P1 fails.

P1 configuration SHA-256 `4c96df20...bb6` and the public gate configuration
SHA-256 `fc687363...7f18` bind the exact split seal, ignored fixture archives,
detector, recognizer, frozen parent, candidate runner, and truth-hidden
evaluator. Candidate execution is not authorized. The public evaluator source
bundle is frozen at `82467dcf...fb15`; it requires direct CPU execution of all
four ONNX streams at three consecutive thresholds, aggregate-only output, and
one canonical gate opening. The public archive remains unopened with zero
evaluations.

The candidate is not reachable from ordinary Auto Detect. It cannot become a
manifest, production-model-store entry, package payload, approved model, or
release input without direct selection, one-time public, marker-composition,
provenance, packaging, clean-machine, and private-validation evidence.

Synthetic fixtures are training and public-test inputs only and are never
application graph data.
