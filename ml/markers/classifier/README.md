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

Production repair revision status:

The `marker-classifier-production-repair-v1` budget is exhausted. Its retained
CLI refuses before creating output. A future authorized repair requires a new
preregistered revision and cannot reuse `P1`, `P2`, or `P3`.

The 2026-08-04 three-candidate budget passed validation and all confirmation
classification metrics, but failed direct packed ONNX parity at the strict
`1e-5` gate. The candidate remains rejected and the manifest remains
fail-closed. That historical confirmation was a same-family repeat and is not
generalization evidence. The disjoint v2 confirmation was then evaluated once
and also failed packed ONNX parity at `1.1444091796875e-05` against the
`1e-05` limit.

`marker-classifier-production-runtime-repair-v2` is a separate preregistered
three-candidate defect revision. P1 is authorized but not yet executed. It
keeps the exact selected checkpoint unchanged and changes only the transport
contract from high-magnitude logits to calibrated probabilities. The full
fixed validation diagnostic preserved every shape, fill, and artifact decision
and measured maximum CPU ONNX error `2.4437904357910156e-06`. New public-v3
and disjoint confirmation-v3 procedural families, manifests, evaluator source
hashes, and strict `1e-5` parity gates are frozen before P1 execution. Neither
gate is authorized until P1 passes selection.

The benchmark command is intentionally single-use for one output directory.
The exact selected weights and evidence remain local until the separately owned
model manifest, C# runtime, packaging, and maintainer gate are complete.
