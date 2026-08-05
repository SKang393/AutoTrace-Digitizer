<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Production model matrix

This matrix records the Goal 19 production decision from repository evidence.
`Candidate` does not mean installed, executable, benchmark-approved, or safe to
bundle. Production automatic detection may use only `Approved` assets. The
marker classifier is the only approved production model file; the required
offline default set remains incomplete.

| Stage | Candidate identity | Exact artifact evidence | Local payload | Benchmark evidence | Decision | Production blocker |
| --- | --- | --- | --- | --- | --- | --- |
| Super-resolution | `RealESRGAN_x2plus` `v0.2.1` | Official payload reverified at SHA-256 `49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb`; BSD-3-Clause notice tracked | Ignored audit copy only | `blocked_adapter_incompatible`; the current NCNN adapter correctly rejects the PyTorch checkpoint | Candidate, blocked | Approve a checksum-bound compatible adapter or artifact, then run all fixed graph metrics. No conversion is authorized. |
| Super-resolution | `realesr-animevideov3` `v0.2.5.0-ncnn-x2` | Official Windows package SHA-256 `abc02804e17982a3be33675e4d471e91ea374e65b70167abc09e31acb412802d`; release-minimal runtime retained only the exact executable, release OpenMP DLL, and x2 param/bin | Ignored audit copy only | `partial_runtime_only`; 2/2 real-adapter Vulkan runs produced exact 2x dimensions in 2366.1442 ms and 1046.5886 ms; cache hit 73.0172 ms; `production_approval: false` | Experimental, failed | Measure marker, OCR, axis, hallucination, and peak-memory metrics; add full NCNN/libwebp/dirent/runtime notices; source `vcomp140.dll` from an authorized Microsoft redistributable; prove CPU fallback or document a compliant required GPU boundary. |
| OCR text detection | `PP-OCRv5_mobile_det` official archive candidate | PaddleOCR `v3.5.0` pinned to `33cbdd9deb2e00f61e7966db70669b249c005a37`; exact 4,935,680-byte archive SHA-256 `50446e5d01ac2a73d5319c89513281f6578414c888c602f9af13f93feefffc58`; official `PaddlePaddle/PP-OCRv5_mobile_det@0d63e78e2b680928f6b1747d76a08db6e645efb7` model card scopes Apache-2.0 to three byte-identical payloads | Ignored audit copy only | Conversion is now license-cleared; no ONNX, parity, provider, or graph benchmark evidence exists yet | Candidate, eligible for conversion only | Perform checksum-bound conversion, source/ONNX parity, CPU execution, frozen public accuracy and exclusion gates, notice review, manifesting, and packaging discovery. |
| OCR text recognition | `en_PP-OCRv5_mobile_rec` official archive candidate | Same pinned source revision; exact 8,007,680-byte archive SHA-256 `e595b4cf2ffad19fbb5a61ba345d63939577a3ab8717b6e5995642590c9101b4`; official `PaddlePaddle/en_PP-OCRv5_mobile_rec@267c36e24c331595590fe7bd72bde2436fd286f2` model card scopes Apache-2.0 to three byte-identical payloads | Ignored audit copy only | Conversion is now license-cleared; no ONNX, parity, provider, or graph benchmark evidence exists yet | Candidate, eligible for conversion only | Perform checksum-bound conversion, source/ONNX parity, CPU execution, frozen public accuracy and role gates, notice review, manifesting, and packaging discovery. |
| Graph numeric recognition | Goal 19 project-trained CTC experiment, seed `20260803` | Original Apache-2.0 procedural 5 by 7 vector glyph corpus; repair ONNX SHA-256 `a48d640226fd95aa67316837abd5a8d08258320b042a5b6a24ea32ee1ab6aa91`; no manifest because quality failed | Ignored failed experiment only | Validation exact `0.03125`, held-out exact `0.015625`, CER `0.932710`; CPU/DirectML maximum difference `1.90735e-05` | Experimental, failed | Experiment budget is exhausted. A new preregistered model must meet maintainer-approved quality gates before manifesting, private validation, or packaged discovery. |
| Graph numeric recognition | Goal 19 dense spatial-sequence V2, seed `20260804` | Original Apache-2.0 procedural 5 by 7 vectors with fully disjoint renderer, glyph-family, and degradation splits; repair ONNX SHA-256 `e7f31d5065f92be6142cd1c17814364646e35d612d5bf0750ed3e403b3b08e3c`; no manifest because quality failed | Ignored failed experiment only | Repair validation exact `0.19921875`, CER `0.855932`; held-out exact `0.10546875`, CER `1.201399`; CPU 6.8874 ms, DirectML 122.6954 ms, maximum difference `6.67572e-06` | Experimental, failed | The two-run V2 budget is exhausted. Do not manifest or enable it; a future preregistered approach needs substantially better cross-family transfer and maintainer-approved quality gates. |
| Graph numeric recognition | Project numeric V1 Candidates 1 through 3 | Apache-2.0 project-owned procedural graph labels only; source seals and reports are checksum-bound; no ONNX was exported because validation failed first | Ignored failed reports and checkpoints only | Validation exact `0.3359375`, `0.34375`, and `0.408203125`; Candidate 3 CER `0.2973821989528796`, role `0.98125`, exclusion `1.0`; later sealed metrics remained unopened | Experimental, failed | The fixed three-candidate budget is exhausted. Do not rerun, recover, tune, manifest, or package these candidates. |
| Marker center detection | `graph-marker-center` `0.1.0` | Original project ONNX SHA-256 `061a496167382d1bd11bb580bed383d2d1725da2001f9c440b7f1acc59ac116a` reproduced byte for byte; Apache-2.0, CPU, and DirectML metadata retained | Ignored audit copy only | Validation F1@5px `1.0000`, zero duplicates and hard-negative hits, and CPU parity pass; historical held-out exact-count gate remains failed at 5/6 fixtures | Experimental, failed | Pass a new preregistered held-out raw-mask gate, private validation, and packaged discovery without reusing the exposed split. |
| Marker shape/fill classification | `graph-marker-classifier` `0.1.0` | Project-owned probability-packed ONNX SHA-256 `26f9304f1689053a0b94aa896a1e239f6ade1e5c1920736a3535c1b32f803b8a`; Apache-2.0 manifest and notice; CPU and DirectML | Checksum-bound ignored production model store | Public-v3 shape `0.9907246376811594`, fill `0.9440313111545988`, artifact and minority F1 `1.0`; disjoint confirmation-v3 passed all classification gates; CPU/DirectML difference `3.5762786865234375e-07`; package resolver and C# probability decoder passed | Approved for production discovery and packaging | Marker stage remains unavailable until an approved marker-center model is present. |

## Deterministic stages

Panelization, axis geometry, legend reasoning, and phase reasoning do not require
model weights. They remain unavailable in the production chain when their
native provenance or upstream evidence dependencies are unavailable. The
OpenCvSharp axis provider specifically remains release-blocked by the native
linked-library audit. The minimal source DLL and linker map now reproduce
byte-for-byte, all 15 evidence entries have candidate dispositions, and four
notice sections validate. Five Microsoft static-runtime entries still require
maintainer attestation, so no reviewed public inventory or replacement runtime
is approved.

## Source records

- `models/manifest/super-resolution/*.json`
- `models/manifest/super-resolution/PROVENANCE_AUDIT.md`
- `models/manifest/ocr/PROVENANCE_AUDIT.md`
- `models/manifest/ocr/GRAPH_NUMERIC_CTC_EXPERIMENT_AUDIT.md`
- `models/manifest/ocr/GRAPH_NUMERIC_SEQUENCE_V2_EXPERIMENT_AUDIT.md`
- `models/manifest/markers/graph-marker-center-0.1.0.json`
- `models/manifest/markers/PROVENANCE_BLOCKER_AUDIT.md`
- `models/manifest/markers/graph-marker-classifier-0.1.0.json`
- `models/manifest/markers/MARKER_CLASSIFIER_CANDIDATE_AUDIT.md`
- `packaging/common/release-audit.json`
- `packaging/opencv-source/review/source-build-review-policy.json`

No model payload, private image, generated weight, or training dataset is
tracked by this matrix. The approved classifier payload remains ignored and is
consumed only through its checksum-bound local production model store.

The Windows build and artifact verifier support multi-file param/bin payloads,
validate each payload against `preprocessing.model_payload_sha256`, and require
exactly one approved offline default for OCR detection, OCR recognition, marker
center, and marker classification. Failed or rejected research manifests do
not become release payload requirements.
