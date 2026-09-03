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

- **Status:** resolved
- **Resolution:** AGENTS.md Section 7.4 prohibits a new gate from rejecting a
  previously approved artifact. The approved payload retains its existing
  `0.90` fill gate; `0.95` is the target for future candidates.
- **Blocks:** none
- **Effort:** complete
- **Action:** no maintainer action remains
- **Why:** public-v3 fill accuracy is `0.9444444444444444`; enforcing the new
  `0.95` bar would retroactively reject a production-approved artifact, which
  AGENTS.md Section 7.4 prohibits
- **Files:** [acceptance bars](../ml/policy/acceptance-bars.json),
  [Phase 3 rescore](../ml/policy/goal22-phase3-rescore-result.json),
  [current approval evidence](../artifacts/production-model-store/evidence/graph-marker-classifier/0.1.0/marker-classifier-production-approval.json)
- **Unblocks when:** resolved by the authoritative compatibility rule

## REV-003 — reauthenticate GitHub as SKang393

- **Status:** open
- **Blocks:** pushing completed Goal 22 phase commits to `origin/main`
- **Effort:** about 2 minutes
- **Action:** sign in to GitHub on this computer as `SKang393`, then confirm
  `gh auth status` reports that account as valid
- **Why:** the active cached `XpressPeach` credential was denied access to
  `SKang393/Graph-auto-reader`, while both GitHub CLI account tokens report as
  invalid; the local repository identity, branch, and remote are correct
- **Files:** [Goal 22 readiness](1.0-READINESS.md),
  [review queue](REVIEW_QUEUE.md)
- **Unblocks when:** `git push origin main` succeeds and `origin/main` matches
  the local `main` commit
