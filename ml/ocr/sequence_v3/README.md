# Canonical-slot graph numeric OCR V3

This experiment isolates glyphs with deterministic pixel geometry, packs them
into canonical slots, and trains a compact slot classifier. It uses only
project-generated procedural labels. Generated corpora, reports, checkpoints,
and ONNX files remain ignored.

The three-candidate V3 budget is exhausted. The training entry point now rejects
every invocation before creating output. This command is retained only as an
executable proof of the guard:

```powershell
python -m ml.ocr.sequence_v3.train `
  --candidate-id candidate-c `
  --output ml/ocr/sequence_v3/runs/prohibited-rerun
```

It raises `ProtocolViolation` because reruns and fourth candidates are
prohibited. Historical Candidate B and C test results are repeatedly observed,
nonsealed research results and cannot support promotion.
