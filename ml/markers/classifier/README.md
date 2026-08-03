# Marker shape, fill, artifact, and embedding classifier

This directory contains an original, deterministic PyTorch training toolchain
for classifying fixed 32 by 32 marker-centered ink-probability patches. It uses
only procedural project data. It does not use private figures, external data,
pretrained weights, downloaded weights, or earlier application source.

## Output contract

`CompactMarkerClassifier` has one compact spatial encoder and four independent
training outputs:

- `shape_logits`: circle, square, triangle-up, triangle-down, diamond, star,
  asterisk, cross, or other;
- `fill_logits`: filled, open, or unknown;
- `artifact_logit`: probability evidence for a non-marker crop;
- `embedding`: a 12-value L2-normalized compact identity vector.

Shape and fill are never collapsed into one class. The embedding receives a
supervised metric loss for matching the same shape/fill symbol through
degradation.

The C# inference adapter accepts one named ONNX output, so `export.py` wraps the
frozen model without changing its weights and concatenates the four training
heads into `classification_heads`, shaped `[N,25]`. Runtime column order is
exactly nine temperature-scaled shape logits, three temperature-scaled fill
logits, one artifact logit, and twelve embedding values. Temperatures are read
from the selected checkpoint, so the C# decoder's ordinary softmax produces the
validated calibrated probabilities. The batch dimension is dynamic and the
spatial input remains fixed at `[N,1,32,32]`. Export verification checks both
whole-tensor parity and exact correspondence between every packed slice and its
documented runtime transform.

## Fixed data and experiment protocol

Train, validation, and held-out families and templates are disjoint. The
procedural corpus covers all required shape/fill combinations, mixed-series
edge context, line-contact deformation, minority probes, and text, axis, tick,
divider, arrow, bracket, intersection, and legend artifacts. Dataset records
include exact tensor hashes. Generated patches, checkpoints, ONNX files, and
reports remain ignored.

`ACCEPTANCE_GATE.md` preregisters a session-local 0.90 macro-F1 gate separately
for shape and fill. It explicitly does not claim maintainer agreement. Three
validation-only model experiments are recorded in `EXPERIMENT_COMPARISON.md`.
The final held-out command creates an exclusive seal and refuses a rerun. A
failure is preserved without test-set tuning.

## Commands

```powershell
python -m pytest ml/markers/classifier/tests -q
python -m ml.markers.classifier.train --output ml/markers/classifier/artifacts/session11-final-e3
python -m ml.markers.classifier.export --checkpoint ml/markers/classifier/artifacts/session11-final-e3/marker-classifier.pt --output ml/markers/classifier/artifacts/session11-runtime-packed/marker-classifier-packed.onnx --report ml/markers/classifier/artifacts/session11-runtime-packed/onnx-parity.json
python -m ml.markers.classifier.benchmark --checkpoint ml/markers/classifier/artifacts/session11-final-e3/marker-classifier.pt --onnx ml/markers/classifier/artifacts/session11-final-e3/marker-classifier.onnx --output ml/markers/classifier/artifacts/session11-final-e3/benchmark.json
```

The benchmark command is intentionally single-use for one output directory.
The exact selected weights and evidence remain local until the separately owned
model manifest, C# runtime, packaging, and maintainer gate are complete.
