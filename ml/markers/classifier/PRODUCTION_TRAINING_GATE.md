# Marker-classifier production training preregistration

This validation-only repair budget was frozen after the first independent public
gate failed shape macro-F1 and before any replacement candidate was trained.
The failed public split is not reused for selection or confirmation.

Exactly three candidates are permitted. All use the existing procedural
training and validation splits, the same compact spatial architecture, and no
private or downloaded data:

| Candidate | Epochs | Learning rate | Weight decay |
|---|---:|---:|---:|
| P1 | 60 | 0.0025 | 0.0001 |
| P2 | 80 | 0.0020 | 0.0001 |
| P3 | 100 | 0.0015 | 0.00005 |

Selection uses validation shape macro-F1, then fill macro-F1, then artifact F1.
A candidate must reach shape and fill macro-F1 of at least `0.90`, artifact F1
of `1.0`, and minimum star/asterisk/cross F1 of at least `0.90` before it can be
exported for a separately frozen confirmation gate.
