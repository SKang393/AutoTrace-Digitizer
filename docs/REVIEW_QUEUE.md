<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Maintainer review queue

Items stay here until resolved. Newest items are appended last.

## REV-001 — supply additional real graphs

- **Status:** open
- **Blocks:** Phase 4 acceptance and the real-image portion of Phase 5
- **Effort:** ~10 minutes
- **Action:** drop at least 4 additional, preferably 10 to 20, representative
  single-case design graph images into
  `artifacts/private-acceptance/inbox/`
- **Why:** the current private set contains one image; Goal 22 requires at least
  5 before private confirmation or production approval. Phase 4R has already
  repaired the measured synthetic envelope and identified a runner-path mismatch.
- **Files:** [private inbox](../artifacts/private-acceptance/inbox/),
  [current aggregate report](GOAL-22-PHASE-4-PRIVATE-ACCEPTANCE.json),
  [Phase 4R gap report](GOAL-22-PHASE-4R-GENERATOR-GAP.json),
  [golden-case criteria](../GOLDEN_CASE_CHANDLER_SPEC.md)
- **Unblocks when:** the inbox and existing private set contain at least 5
  maintainer-supplied graph images total
