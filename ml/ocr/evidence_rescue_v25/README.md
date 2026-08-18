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

P1 performs zero optimizer steps. The train split is still frozen to preserve
the complete three-way split identity and may support a later separately
preregistered candidate if P1 fails. Validation and truth-hidden public
families, seed offsets, renderer identifiers, degradation identifiers, and
fixture bytes are fresh and disjoint. The public archive remains locked until
a candidate passes the fixed visible-selection gates.

The candidate is not reachable from ordinary Auto Detect. It cannot become a
manifest, production-model-store entry, package payload, approved model, or
release input without direct selection, one-time public, marker-composition,
provenance, packaging, clean-machine, and private-validation evidence.

Synthetic fixtures are training and public-test inputs only and are never
application graph data.
