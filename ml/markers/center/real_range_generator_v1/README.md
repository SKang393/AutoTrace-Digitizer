# Real-range marker input generator

This synthetic-only diagnostic generator repairs the marker-center input gap
identified by Phase 4R. It reuses the project-owned procedural marker and
artifact primitives, uses disjoint deterministic train/dev seeds, and spans
effective marker diameters from 1 through 48 px. Its aggregate quantiles are
6 px at p05, 12 px median, 24 px at p90, and 27 px at p95.
The rendered footprint contract is measured deterministically from the marker
primitive, including the one-pixel point and the 48 px target (49 px raster
span where PIL's inclusive ellipse box requires it).

Each split contains 2,004 truth centers. Exactly 75 centers overlap the OCR
hard-mask window and 332 overlap the artifact hard-mask window at the 0.35
threshold, with the remainder serving as unmasked controls. Text,
line-intersection, and axis negatives are included. Axis-like horizontal and
vertical crossings cover artifact patch topology; the first OCR overlap is a
full text-region-like patch. Only aggregate statistics
and hashes are written; no pixels, truth rows, scene identifiers, private
corpus data, model, or training path is used.

Both disjoint splits use the same deterministic marker family: two modes keep
the existing project marker primitive unchanged, three add filled overlays,
and two use filled elongated overlays. This increases center occupancy, row
support, and ring support while retaining hollow-marker cases. The generator also includes
deterministic anti-aliased topology negatives:
off-center multi-branch junctions and elongated fragments. The negative
proposal audit records their morphology envelopes and checks that the real-dev
medians for dark fraction, center occupancy, row and column support, extent
balance, covariance ratio, border support, and ring support are represented.
These checks are aggregate input-coverage checks, not model-selection rules.

A separate negative-proposal audit runs the exact V24 mask-preserving ink-supported proposal
extractor and labels proposals positive only when their centers are within 3 px
of a truth center. It reports full train/dev proposal, positive, and negative
counts plus aggregate quantiles for ink, OCR-mask, and artifact-mask proposal
patch features. Positive morphology gates use the central p25 to p75 envelope
of both train and dev. The train-only sampler retains every declared topology
junction and fragment proposal within 16 px, then retains the nearest eligible
negative proposal within 4 px of the one-third and two-thirds anchors on every
consecutive truth-center connector. It fills the existing generic quota to keep
exactly 32,580 negatives and fails closed when an anchor has no nearby eligible
proposal. The audit records anchor coverage, selected counts, and aggregate
selection hashes.

Run from the repository root:

```powershell
python -m ml.markers.center.real_range_generator_v1.audit --output ml/markers/center/real_range_generator_v1/AUDIT.json
python -m ml.markers.center.real_range_generator_v1.negative_proposal_audit --output ml/markers/center/real_range_generator_v1/NEGATIVE_PROPOSAL_AUDIT.json
python -m pytest ml/markers/center/real_range_generator_v1/tests -q
```
