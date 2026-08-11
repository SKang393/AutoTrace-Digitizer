# Graph text stride-4 detector V4

This revision addresses one defect observed after the three-candidate
`graph-text-ignore-band-v3` budget was exhausted. V3 P3 removed every
selection-set exclusion false positive, but it missed 19 of 80 text fixtures
after an eightfold spatial bottleneck. V4 P1 changes only the detector
topology to preserve fourfold spatial detail, adds one shallow fine-detail
skip, and retains the frozen P3 objective, optimizer-step budget, production
DB postprocessing, and fail-closed gates.

The train, visible selection, and truth-hidden public splits use new procedural
renderer, degradation, structure, and seed families. They contain no Chandler
image, article image, private image, downloaded data, prior selection fixture,
prior sealed fixture, or `Generalization` label. The public archive may not be
opened unless a committed candidate passes every selection fixture with zero
false regions, duplicates, and prohibited exclusion hits plus CPU ONNX parity.

Candidate P1 ran exactly once and failed the visible selection gate. P2 is
preregistered to replace only its transposed-convolution upsampling with
bilinear resize-convolution. Neither candidate is a production model. Both
remain ineligible for a manifest, model-store discovery, packaging, production
approval, and release approval until all later direct-evidence gates pass. The
deterministic axis, tick, and divider masking used by these procedural detector
inputs also does not prove that an approved production artifact-mask provider
exists. That production workflow blocker remains mandatory and fail closed.
