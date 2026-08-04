<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

# PaddleOCR model candidate provenance audit

Audit date: 2026-08-03

Goal 19 also exercised the project-owned graph-numeric training path. That
bounded experiment produced a runtime-compatible but quality-failed ONNX and
did not create a model manifest. See
`GRAPH_NUMERIC_CTC_EXPERIMENT_AUDIT.md`. Text detection and general text
recognition remain separately blocked by the PaddleOCR evidence below.

## Scope and decision

This directory contains metadata only. No Paddle model archive, unpacked model,
converted ONNX file, dictionary, runtime binary, test image, or benchmark output
is included.

The official PaddleOCR documentation identifies two compact candidates that fit
the intended local OCR split:

| Task | Candidate | Reason for evaluation |
|---|---|---|
| Text detection | `PP-OCRv5_mobile_det` | Official mobile text detector intended for efficient edge deployment |
| English and numeric recognition | `en_PP-OCRv4_mobile_rec` | Official lightweight recognizer explicitly described as supporting English and numeric characters |

Both candidates are **unbundled and release-blocked**. The official repository
license and ONNX conversion documentation are useful evidence, but they do not
establish an immutable, checksum-pinned model artifact with an explicit
artifact-level redistribution statement. No schema manifest is created because
`contracts/model-manifest.schema.json` requires exact SHA-256, file, license,
commercial-use, redistribution, input/output, and provider assertions.

## Pinned official repository evidence

- Repository: <https://github.com/PaddlePaddle/PaddleOCR>
- Documentation tag: `v3.5.0`
- Documentation commit: `33cbdd9deb2e00f61e7966db70669b249c005a37`
- Root license at that commit:
  <https://github.com/PaddlePaddle/PaddleOCR/blob/33cbdd9deb2e00f61e7966db70669b249c005a37/LICENSE>
- Root license Git blob: `430edfa0cfcb59f15b7aa5457427d0faf2e40dac`
- Root license raw size: `11,376` bytes
- Root license raw SHA-256:
  `3840c5c0c61c294264d2dd77b8777be6ddd90121ef4e0e64abcd22edea581d6e`
- Root repository license: Apache-2.0
- GitHub release `v3.5.0` contains no downloadable model assets.

The root license establishes the license of the pinned repository contents. It
does not by itself prove that separately hosted model archives carry the same
terms. No artifact-specific license or explicit redistribution grant for the
two archives was located in the pinned official model documentation.

## Detection candidate

### Identity

- Candidate: `PP-OCRv5_mobile_det`
- Task: text-region detection
- Official documentation at the pinned commit:
  <https://github.com/PaddlePaddle/PaddleOCR/blob/33cbdd9deb2e00f61e7966db70669b249c005a37/docs/version3.x/module_usage/text_detection.en.md>
- Documentation Git blob: `64546c4a20fddb08a2ec6225cc245c7b180ed97d`
- Official inference archive URL:
  <https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar>

The pinned documentation describes this as a mobile detector for efficient
edge deployment. Its table reports a 4.7 MB model storage size and detection
Hmean of 79.0 on the upstream benchmark. Those upstream figures are selection
metadata only, not Graph Auto Reader acceptance results.

### Non-downloading endpoint check

A metadata-only HTTP `HEAD` request on 2026-08-02 returned:

- Status: `200`
- `Content-Length`: `4,935,680`
- `Last-Modified`: `Mon, 19 May 2025 20:17:44 GMT`
- `ETag`: `"-cd090f2a008766d8fe7cdba8d15a38ea"`
- `Content-Type`: `application/octet-stream`
- Provider-reported CRC32: `447818556`

The HTTP ETag is not treated as a SHA-256 checksum or immutable revision.

## Recognition candidate

### Identity

- Candidate: `en_PP-OCRv4_mobile_rec`
- Task: English and numeric text recognition
- Official documentation at the pinned commit:
  <https://github.com/PaddlePaddle/PaddleOCR/blob/33cbdd9deb2e00f61e7966db70669b249c005a37/docs/version3.x/module_usage/text_recognition.en.md>
- Documentation Git blob: `a52c71a09116d4da09b6a2b4eaff500e1d9849d4`
- Official inference archive URL:
  <https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv4_mobile_rec_infer.tar>

The pinned documentation explicitly describes this candidate as supporting
English and numeric character recognition. Its table reports a 7.5 MB model
storage size and recognition accuracy of 70.39 on the upstream benchmark.
Those upstream figures are selection metadata only. They do not satisfy the
required graph-specific synthetic numeric exact-match benchmark.

### Non-downloading endpoint check

A metadata-only HTTP `HEAD` request on 2026-08-02 returned:

- Status: `200`
- `Content-Length`: `7,833,600`
- `Last-Modified`: `Thu, 05 Jun 2025 13:45:17 GMT`
- `ETag`: `"-6a258098c2bacf7b2fc4add9327ad88e"`
- `Content-Type`: `application/x-tar`
- Provider-reported CRC32: `1715592068`

The HTTP ETag is not treated as a SHA-256 checksum or immutable revision.

### Recognition dictionary contract

The pinned training configuration binds this candidate to
`ppocr/utils/en_dict.txt` and enables spaces. The dictionary order is part of
the output-index contract and cannot be replaced with a guessed numeric
alphabet.

- Configuration:
  <https://github.com/PaddlePaddle/PaddleOCR/blob/33cbdd9deb2e00f61e7966db70669b249c005a37/configs/rec/PP-OCRv4/en_PP-OCRv4_mobile_rec.yml>
- Dictionary:
  <https://github.com/PaddlePaddle/PaddleOCR/blob/33cbdd9deb2e00f61e7966db70669b249c005a37/ppocr/utils/en_dict.txt>
- Dictionary Git blob: `7677d31b9d3f08eef2823c2cf051beeab1f0470b`
- Dictionary SHA-256:
  `f27a6aa993c9cb67a588e7ea9aea90bb96b8e51dec6ce98bd7e76c104c1829fe`

PaddleOCR export code serializes the configured character dictionary into the
inference configuration. The separately hosted archive remains uninspected, so
this source dictionary cannot yet be asserted as its exact deployed mapping.

## ONNX suitability

The official PaddleOCR deployment guide states that PaddleOCR static-graph
models can be converted to ONNX with the Paddle2ONNX plugin provided through
PaddleX:

- Pinned guide:
  <https://github.com/PaddlePaddle/PaddleOCR/blob/33cbdd9deb2e00f61e7966db70669b249c005a37/docs/version3.x/deployment/obtaining_onnx_models.en.md>
- Documentation Git blob: `8622a03ec335dc216721b48d8cc5868b6527632f`

This establishes a documented conversion path, not a verified Graph Auto
Reader artifact. Neither candidate was downloaded, converted, inspected with
ONNX tooling, loaded with ONNX Runtime, or benchmarked in this metadata-only
work. CPU and DirectML compatibility therefore remain unverified.

## Required local audit fields

| Dependency/model | Version | Source | License | Bundled or downloaded | Notice path | Checksum | Review status |
|---|---|---|---|---|---|---|---|
| `PP-OCRv5_mobile_det` | Artifact revision not published as an immutable identifier | Official PaddleOCR v3.5.0 documentation and separately hosted inference archive | Repository is Apache-2.0; archive terms not proven | Neither | Missing approved model notice | Not measured | Candidate only; release blocked |
| `en_PP-OCRv4_mobile_rec` | Artifact revision not published as an immutable identifier | Official PaddleOCR v3.5.0 documentation and separately hosted inference archive | Repository is Apache-2.0; archive terms not proven | Neither | Missing approved model notice | Not measured | Candidate only; release blocked |

## Release blockers

Each candidate remains blocked until all of the following are complete:

1. Obtain an explicit official license and redistribution statement applicable
   to the exact model archive and any required dictionary.
2. Pin an immutable upstream artifact revision or preserve an approved source
   archive under the project's external-artifact policy.
3. Download only into approved temporary audit storage and record the source
   archive size and SHA-256.
4. Inventory and hash every required internal file.
5. Convert with pinned PaddleX and Paddle2ONNX versions and record the exact
   command, opset, converter versions, output files, and ONNX SHA-256 values.
6. Inspect the converted input/output tensors and record exact preprocessing,
   postprocessing, dictionary, coordinate-space, and provider contracts.
7. Validate CPU and DirectML inference in the Windows runtime.
8. Run the fixed synthetic graph OCR benchmark, including small, faded,
   rotated, decimal, percent, negative, `0/O`, and `1/l` cases.
9. Add the complete notice set and a schema-valid manifest only after the
   preceding evidence supports every required field.

## Privacy and Git eligibility

The evidence references public official upstream documentation and endpoints.
It contains no article data, private research images, human annotations,
weights, or generated training data. This audit document is Git-eligible.

## Verification method

The official GitHub API was queried for the `v3.5.0` tag, documentation blobs,
and root license blob. The raw root license was hashed in memory. Artifact
endpoints were checked with HTTP `HEAD` requests only. No response body for a
model archive was requested, persisted, unpacked, or added to the repository.
