<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Phase divider synthetic held-out fixture

- Source: created in the clean Graph Auto Reader repository for Session 13 from deterministic synthetic segment geometry. It does not originate from a previous project or an article.
- Purpose: fixed evaluation of dashed-segment aggregation error in original pixel and session-pitch units. It does not measure upstream pixel-level line extraction or real-article performance.
- Fixture: `phase-divider-synthetic-heldout-v1.csv`
- SHA-256: `a6f47ae54bba77adee7bada7cc5d77328f820389945db6c4ea8b8d658430fa1e`
- Copyright and license: Copyright 2026 Sungwoo Kang, Apache-2.0.
- Privacy status: synthetic only, with no private or human-subject data.
- Git eligibility: eligible test fixture and provenance documentation.
- Freeze rule: changing any coordinate, expected value, or session pitch requires a new fixture version and checksum.

The 20 cases contain fully materialized segment coordinates. Tests load these
coordinates directly and do not derive inputs from the expected divider x.
