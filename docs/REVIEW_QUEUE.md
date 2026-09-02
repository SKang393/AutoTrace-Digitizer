<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Maintainer review queue

Items stay here until resolved. Newest items are appended last.

## REV-001 — supply additional real graphs

- **Status:** resolved
- **Resolution:** the authoritative `data/manual data/` corpus was identified
  on 2026-09-02 with 40 studies, 171 complete `.dig` projects, and 3,055
  digitized points. The earlier five-image request is superseded.
- **Blocks:** none
- **Effort:** complete
- **Action:** no maintainer action remains; Phase 4 now owns the study-level
  `real-dev` and `real-sealed` harness.
- **Why:** the complete local corpus replaces the one-image acceptance inbox.
- **Files:** [real corpus](../data/manual%20data/),
  [historical aggregate report](GOAL-22-PHASE-4-PRIVATE-ACCEPTANCE.json),
  [golden-case criteria](../GOLDEN_CASE_CHANDLER_SPEC.md)
- **Unblocks when:** resolved by the local corpus inventory

## REV-002 — resolve marker fill gate compatibility

- **Status:** open
- **Blocks:** Goal 22 Tier 1 completion and production-model promotion
- **Effort:** ~5 minutes
- **Action:** confirm whether the already approved marker-classifier payload
  keeps its existing `0.90` fill gate while `0.95` becomes the target for a
  future candidate, or authorize a different compatible rule
- **Why:** public-v3 fill accuracy is `0.9444444444444444`; enforcing the new
  `0.95` bar would retroactively reject a production-approved artifact, which
  AGENTS.md Section 7.4 prohibits
- **Files:** [acceptance bars](../ml/policy/acceptance-bars.json),
  [Phase 3 rescore](../ml/policy/goal22-phase3-rescore-result.json),
  [current approval evidence](../artifacts/production-model-store/evidence/graph-marker-classifier/0.1.0/marker-classifier-production-approval.json)
- **Unblocks when:** the maintainer records a compatible current-payload gate
  and future-candidate target
