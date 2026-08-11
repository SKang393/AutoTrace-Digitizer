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

P1 ran exactly once. It passed CPU ONNX parity but failed validation at exact
match `0.8515625`, role accuracy `0.8645833333333334`, and exclusion accuracy
`0.890625`. Its sealed-public archive remains unopened.

P1 validation showed one isolated representation defect: max-fit normalization
discarded absolute component scale. It mapped 31 of 111 decimal points to the
reject class and mapped 14 full-height divider exclusions to digit one at the
best threshold. P2 is preregistered to replace only that normalization with an
absolute-scale, full-label-height encoding. The frozen rasters, MLP, loss,
optimizer, epochs, thresholds, gates, and still-hidden public split are
unchanged. P2 may run only after its exact sources, configuration, result
trigger, and canonical budget authorization are committed.

Passing validation selects a research candidate only. Passing the public gate
would still not create a production manifest, approve a model-store payload,
prove detector composition, authorize packaging, or change release readiness.
