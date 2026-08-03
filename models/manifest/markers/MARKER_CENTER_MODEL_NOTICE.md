<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

# Graph Auto Reader marker-center model notice

## Covered artifact

- Model ID: `graph-marker-center`
- Model version: `0.1.0`
- File: `marker-center.onnx`
- Size: `37,542` bytes
- SHA-256: `061a496167382d1bd11bb580bed383d2d1725da2001f9c440b7f1acc59ac116a`
- ONNX opset: `18`

## Copyright and license

Copyright 2026 Sungwoo Kang.

The model source, procedural training data, trained checkpoint, and exported
ONNX weights are original Graph Auto Reader work licensed under Apache License
2.0. The complete license text is in the repository root `LICENSE` file.

Commercial use and redistribution are permitted under Apache-2.0, including
its notice and attribution requirements. This notice must remain with any copy
of the model.

## Provenance

- Training revision: `marker-center-pytorch-v1`
- Architecture: `compact-pplcnet-depthwise-fpn-v1`
- Canonical LF-normalized executable source snapshot SHA-256:
  `a8cfd538b164d0e7a3c19c262daf55413eaea9a7639f3a1bc87d0c341b9734ac`
- Sealed procedural dataset manifest SHA-256:
  `f313278c0fa987ec2b9225b3dbdf3ed218fbe8f5c8d6d32aa339ccad2dc4a6c7`
- Trained checkpoint SHA-256:
  `72471156daec005b7617916a3d81c3966c0a5e36cb93c6c88aaa58f4d1272fb9`
- Evidence index SHA-256:
  `85de0a95bf8ec5e3a94ce928e83d319a4dae476ae1a82e9d144b0c23266c8afd`
- Deterministic no-heldout reproduction evidence SHA-256:
  `91191738ffaee38c145f162c3a619305e1336f0646c85bc3bc2755f5225ee87c`
- Exact dependency-license audit evidence SHA-256:
  `ea1faec18b744113b18d3e60815b29ce65cc057062611ed26f6bb9da85368228`

The canonical source digest is stable across Windows and Unix checkouts. For
the eight named executable files, normalize CRLF to LF, hash each normalized
UTF-8 file, sort lines by file name, concatenate `name=sha` plus LF for every
line, and hash the resulting UTF-8 material.

No pretrained weights, external graph figures, article data, private data,
human annotations, or third-party source were used to produce the model.
PyTorch, ONNX, ONNX Runtime, NumPy, Pillow, and pytest were unbundled training,
export, evaluation, or test tools. Their licenses do not replace this model's
Apache-2.0 license. Exact wheel-extracted dependency license and notice bytes
were independently hash-checked against the local provenance ledger.

## Release status

The exact model is manifested for auditability but is not release-eligible.
Session 10 and model acceptance are `FAIL`.

The preserved held-out report predates the production raw-mask max-gating
parity repair. It is historical artifact-head-only postprocessing evidence,
not production-runtime acceptance evidence. Its unmodified metrics found all
38 true centers with no duplicates and one unrelated false positive, so only 5
of 6 golden fixtures had exact point counts. Its zero-mask diagnostic detected
two legend glyphs.

After the parity repair, deterministic training and export reproduced the
checkpoint, ONNX, and dataset-manifest hashes byte-identically with
`heldout_test_evaluations=0`. No held-out benchmark was rerun. These failed and
unresolved criteria require a new model version and newly authorized held-out
evaluation, not reinterpretation of the historical metrics.
