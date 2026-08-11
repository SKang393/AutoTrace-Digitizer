# Component-Geometric OCR V4

This defect class preregisters a project-owned graph-numeric recognizer after
the earlier CTC, spatial V2, canonical-slot V3, and numeric V1 experiments
failed their fixed gates. It does not reuse their checkpoints or architecture.

The model classifies deterministically isolated glyph components using fixed
grid, row, column, and radial projections followed by a small MLP. It supports
integers, decimals, negatives, percentages, and the registered `O` versus `0`
and `l` versus `1` display ambiguities. It has no convolution layers and uses no
pretrained weights.

The train, validation, and sealed-public renderers use separately reviewed Noto
Sans font files already shipped by the application. The split is synthetic
only. Chandler, private article images, external datasets, and prior project
weights are prohibited.

P1 may run only after the tracked protocol, selection manifest, sealed-public
seal, gate configuration, training configuration, runner sources, and canonical
training-budget authorization are committed. Training reads only train and
validation samples. The ignored sealed-public archive is checksum-verified but
not opened until a later, separately authorized single-use gate.

Passing validation selects a research candidate only. Passing the public gate
would still not create a production manifest, approve a model-store payload,
prove detector composition, authorize packaging, or change release readiness.

