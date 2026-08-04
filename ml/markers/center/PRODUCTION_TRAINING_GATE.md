# Marker-center production training preregistration

This validation-only repair budget was frozen after the first independent public
gate failed exact-count coverage and before any replacement candidate was
trained. The failed public split is not reused for selection or confirmation.

Exactly three candidates are permitted. They use the existing procedural
training and validation scenes, unchanged labels, the same compact FPN model,
and deterministic photometric robustness augmentation only:

| Candidate | Epochs | Learning rate | Robustness mode |
|---|---:|---:|---|
| P1 | 110 | 0.0025 | contrast |
| P2 | 130 | 0.0020 | resample |
| P3 | 150 | 0.0015 | mixed contrast and resample |

Model and threshold selection use standard validation plus deterministic
photometric validation variants. The score prioritizes exact scene count, then
5 px F1, then zero duplicates and hard-negative rejection. No public-gate or
private image result may affect selection.

## Historical execution invalidation

The retained 2026-08-04 P1, P2, and P3 runs are invalid protocol evidence. The
executed implementation also dropped text-mask and artifact-mask channels on a
deterministic schedule, although the preregistration allowed photometric
augmentation only. The implementation has been corrected for any future,
separately authorized experiment, but the three retained runs are not rerun,
reclassified, or approved. Their failed reports remain historical evidence.
