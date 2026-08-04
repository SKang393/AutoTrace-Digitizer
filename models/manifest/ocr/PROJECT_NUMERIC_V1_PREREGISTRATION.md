<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Project numeric OCR V1 preregistration audit

Audit date: 2026-08-04

## Decision

Status is **frozen, untrained, and not approved**. No checkpoint, ONNX, model
manifest, notice, production resolver entry, package payload, or release claim
exists. Training is prohibited until the complete bound source bundle is
reviewed and committed unchanged on `main`.

## Distinct architecture

The candidate is a whole-crop 2D encoder with one global spatial bottleneck,
eight semantic character heads, and a numeric-role head. It does not use CTC
loss, original-column dense alignment, deterministic glyph isolation, or the V3
canonical-slot topology. The semantic logits are projected to fixed
blank-separated runtime positions only after whole-crop classification.

## Frozen procedural evidence

| Split | Numeric | Exclusion | Renderer | SHA-256 |
|---|---:|---:|---|---|
| Train | 4,096 | 512 | independent polyline stroke | `3217f58667ca497e8b5de60b4b493c92efd75e7f2e49b71a4f847ff6b7a5c8d2` |
| Validation | 512 | 128 | independent bitmap | `dec5869ac463b24c41616dafda34eb5817b8392a3da88ce09fd60fb47d7520f3` |
| Sealed test | 512 | 128 | independent seven segment | `5992d05dd37c84a4816617d3789c85a8e2144f419a87d24432304cc804398827` |

The split implementations have separate glyph definitions and degradation
families. Inputs are deterministic procedural graph labels and exclusion shapes
only. No Chandler image, private graph, article data, external dataset, font, or
pretrained weight was read.

## Budget and gates

The defect class allows at most three candidates. Only Candidate 1 has a frozen
configuration. Candidates 2 and 3 remain non-executable reservations requiring
a committed validation-only defect report and a newly frozen one-factor change.
The fixed public gates are validation and sealed exact match `>=0.90`, sealed CER
`<=0.05`, role accuracy `>=0.90`, marker exclusion `1.0`, CPU ONNX parity
`<=1e-4`, decoded equality, and dynamic `[N,32,14]` plus `[N,2]` outputs.
Marker exclusion measures whether procedural nonnumeric crops are rejected. Zero
marker creation from recognized text is a separate application-integration gate
and is intentionally not claimed by the model training report.

## License and provenance

- Source and future project-owned weights: Copyright 2026 Sungwoo Kang,
  Apache-2.0.
- Local training tools: PyTorch BSD-3-Clause, NumPy BSD-3-Clause, ONNX
  Apache-2.0, and ONNX Runtime MIT, unbundled and recorded in
  `ml/ocr/DEPENDENCY_PROVENANCE.csv`.
- Generated data, checkpoints, ONNX, and reports remain ignored and are not Git
  eligible.
- Source, tests, frozen metadata, and this audit are Git eligible after review.

## Remaining blockers

1. Review and commit the exact preregistration and source binding on `main`.
2. Run Candidate 1 exactly once on CPU without configuration changes.
3. Meet every fixed validation, sealed, role, exclusion, CPU, and parity gate.
4. Prove downstream application integration creates no markers from recognized
   text.
5. Complete private graph validation without training or tuning on private data.
6. Prove DirectML, production resolver, notice, packaging, offline, installer,
   portable, and release-audit gates.

Until all mandatory gates pass, no manifest or approval is permitted.
