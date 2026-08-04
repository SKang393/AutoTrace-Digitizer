# Marker-center production repair v2 candidate P2

P1 passed procedural selection but failed the sealed public gate with one short
tick hit and two false-positive centers in every fixture. P2 treats this only
as a general defect class: short perpendicular tick and joint structures.
Public tensors, coordinates, and labels are excluded from selection.

P2 changes one factor. It appends deterministic procedural train and validation
families containing six short perpendicular tick or joint hard negatives per
added scene. Train and validation family names and seeds are disjoint. Original
markers, targets, and procedural selection scenes remain present.

Everything else is frozen to P1:

- compact center architecture;
- seed `20260815` and 130 epochs;
- learning rate `0.002` and weight decay `0.00001`;
- mixed photometric robustness;
- center, radius, artifact, marker-artifact, and mask-consensus loss weights;
- mask-consensus threshold `0.20`;
- raw-mask max-gated postprocessing;
- threshold sweep `0.28`, `0.32`, `0.36`, and `0.40`.

The fixed P2 selection manifest contains 12 training scenes and 6 validation
scenes. Its SHA-256 is
`afb792b635472bc76d950249dd366f6d9c6f71099a39733d0cd0eb9e6722b59a`.
P2 must refuse before output until its dataset, config, runner, budget entry,
and sealed manifest have been reviewed, committed, and verified clean.
