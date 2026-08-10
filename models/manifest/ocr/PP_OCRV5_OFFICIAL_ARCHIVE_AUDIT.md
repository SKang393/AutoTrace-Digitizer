<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

# PP-OCRv5 official archive bake-off audit

Audit date: 2026-08-05

## Decision

The exact source bytes are **eligible for conversion only**. Each BOS inference
archive is byte-identical to the corresponding immutable official PaddlePaddle
model repository, and each exact model card scopes Apache-2.0 to that repository.
The executable audit independently binds owner, model ID, revision, public and
ungated state, complete file inventory, model-card SHA-256, license metadata,
revision-specific Git blob and LFS identities, and all six inference payload
hashes.

The controlled conversion experiment now passes for both exact source models.
Two independent opset-11 conversions per model were byte-identical, the ONNX
full checker passed, and 16 deterministic raw tensor comparisons per model ran
on `CPUExecutionProvider`. The detector ONNX SHA-256 is
`d4aa24d408cd70b8b9f66cc758e20f397fc31a9c69d8477cf8887fc53bd5fceb`
with maximum Paddle-to-ONNX difference `1.349323213162279e-07`. The recognizer
ONNX SHA-256 is
`7839f12b644f574eaf677e92a11bd3e337f4b2f910160666073888783fece743`
with maximum difference `2.205371856689453e-06`. The ignored conversion report
SHA-256 is `8c6c8aa522663c0594894f0b526c9fcc0055139217d8d1b6bac968ed1c4d9d15`.
The later once-only public and sealed OCR benchmark ran and failed every
accuracy, role, detection, and exclusion threshold. No independent composed
marker-stage evidence exists, no approved production manifest exists, and no
payload entered the production model store or packaging. A reviewed Apache-2.0
license and ONNX change-notice bundle now exists under `LICENSES/`, but it is
not yet bound through the model store or either package.

## Pinned official source

- Repository: `https://github.com/PaddlePaddle/PaddleOCR`.
- Tag: `v3.5.0`.
- Tag target and documentation commit:
  `33cbdd9deb2e00f61e7966db70669b249c005a37`.
- Commit tree: `4fd5a734b7a96ce2808c261e89accb90e7299e37`.
- Text-detection documentation blob:
  `64546c4a20fddb08a2ec6225cc245c7b180ed97d`.
- Text-recognition documentation blob:
  `a52c71a09116d4da09b6a2b4eaff500e1d9849d4`.

The pinned documentation does supply both requested models. There is no
tag/commit assertion conflict.

The executable audit independently verifies exact ref
`refs/tags/v3.5.0`, official PaddleOCR ref and commit response URLs, target
commit, commit tree, measured Git blob identities for both documents, and
presence of each exact archive URL in its pinned document. Current result:
`source_provenance_valid=true`.

## Exact official archives

### PP-OCRv5 mobile detection

- Model: `PP-OCRv5_mobile_det`.
- URL: `https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar`.
- Bytes: `4,935,680`.
- SHA-256:
  `50446e5d01ac2a73d5319c89513281f6578414c888c602f9af13f93feefffc58`.
- Last-Modified: `Mon, 19 May 2025 20:17:44 GMT`.
- ETag: `-cd090f2a008766d8fe7cdba8d15a38ea`.
- Provider CRC32: `447818556`.

| Member | Bytes | SHA-256 |
|---|---:|---|
| `inference.json` | 229,777 | `05feef1acb00aa4cd7362b15f7f501fc4f99d7b1fa73c1c871e0c7b1504b0f5c` |
| `inference.pdiparams` | 4,692,937 | `afa1820cb16c1fd0dad589d0f8b389139061c1ef6d68019685fd07be997dda5b` |
| `inference.yml` | 903 | `98069072e1b6b37d727fd9d9f11725faa46d6ea0de012f2ed26caea011c37699` |

The YAML identifies DB postprocessing, BGR input, resize-long `960`, and the
documented normalization constants. It contains no license metadata.

### PP-OCRv5 English recognition

- Model: `en_PP-OCRv5_mobile_rec`.
- URL: `https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv5_mobile_rec_infer.tar`.
- Bytes: `8,007,680`.
- SHA-256:
  `e595b4cf2ffad19fbb5a61ba345d63939577a3ab8717b6e5995642590c9101b4`.
- Last-Modified: `Thu, 21 Aug 2025 03:42:28 GMT`.
- ETag: `-921bbbd48af68f5d7b13a2a787ffb2be`.
- Provider CRC32: `2104810687`.

| Member | Bytes | SHA-256 |
|---|---:|---|
| `inference.json` | 217,712 | `fd1b6ec722ea841a72d3ba43e527df1d1066d5d7808e0503ee3eec7265188753` |
| `inference.pdiparams` | 7,772,315 | `3ec8a97ed6cefe8568d3e2ee90bb193299b566a7661aa4fd52d224b96b59f66b` |
| `inference.yml` | 3,964 | `27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067` |

The YAML binds `en_PP-OCRv5_mobile_rec` to its embedded character dictionary,
CTC decoding, and `[3,48,320]` recognition preprocessing. It contains no
license metadata.

## Exact official model repositories

The official PaddlePaddle Hugging Face repositories provide artifact-scoped
model-card terms for the exact bytes present in the BOS archives:

| Model | Immutable official repository revision | Model-card SHA-256 | License | Byte identity |
|---|---|---|---|---|
| `PP-OCRv5_mobile_det` | `PaddlePaddle/PP-OCRv5_mobile_det@0d63e78e2b680928f6b1747d76a08db6e645efb7` | `4cc20ad6d41af86b3ce9885ffb0956e152574a2eb14179aeb07fd2d3956161ca` | Apache-2.0 | All three inference files match the BOS member SHA-256 values above |
| `en_PP-OCRv5_mobile_rec` | `PaddlePaddle/en_PP-OCRv5_mobile_rec@267c36e24c331595590fe7bd72bde2436fd286f2` | `4c1cfd6e103b0966fe97505b5254cfa35a931d47d7effca97a9db47fb57dd699` | Apache-2.0 | All three inference files match the BOS member SHA-256 values above |

The audit requires author `PaddlePaddle`, exact repository ID and revision,
`private=false`, `gated=false`, the reviewed six-file repository inventory,
exactly one `license: apache-2.0` model-card field, exact model identity, no
contradictory license term, and all three byte hashes for both models. Every
non-LFS file must reproduce its revision-bound Git blob SHA-1. Each weights file
must reproduce the reviewed Git LFS pointer blob, pointer size, content size,
and content SHA-256. The report emits every consumed local evidence path plus
its revision-specific API and file URL. Malformed or extra sibling fields and
any missing, changed, duplicated, gated, or mismatched input fail closed.

## Artifact-level terms result

The complete six-file extracted inventory contains no `artifact-terms.json`,
license, or notice member. A content search also found no license, copyright,
redistribution, notice, commercial-use, or similar grant. Keyword presence
would not be sufficient: the executable gate requires an allowlisted exact SPDX
identifier, explicit boolean-true redistribution and commercial-use grants,
archive-wide scope, exact model identity, and a reviewed `NOTICE` path whose
SHA-256 and applicability lines match the structured terms. Proprietary, GPL,
AGPL, SSPL, BUSL, noncommercial, research-only, commercial-forbidden, or
redistribution-forbidden notice content is rejected even when the structured
fields claim permission. The decoded notice must contain exactly four unique
nonempty lines matching the scoped model ID, allowlisted SPDX license,
redistribution grant, and commercial-use grant. Any extra prose, duplicate,
condition, changed spelling, or homoglyph fails closed. Non-notice paths are
rejected. The decision also requires exactly one full-metadata audit for each
pinned candidate, with no missing, altered, extra, or duplicate audit.
Duplicate JSON object keys are rejected at every depth for artifact terms and
saved source-provenance responses. Duplicate normalized tar member paths are
rejected before payload reads and inventory construction, preventing last-entry
or dictionary-collapse behavior.
Windows-target path review additionally requires canonical POSIX member
spelling, Windows-safe segments, and case-insensitive uniqueness. Backslashes,
repeated separators, absolute paths, dot segments, trailing dots or spaces,
colons or alternate data streams, invalid Windows characters, reserved device
names, and case-only collisions all block the archive before extraction.
Archive SHA-256 validation occurs before opening the tar. The complete member
table then permits only regular files and canonical safe directories, validates
directory paths, and enforces one Windows-equivalent identity across every
member. Symbolic and hard links, devices, FIFOs, sparse and contiguous entries,
and unknown types are rejected before payload extraction.
The metadata pass also rejects every nonempty `TarInfo.sparse` map regardless of
the member type byte, including PAX GNU sparse metadata attached to a nominal
regular `b'0'` member.

The BOS tar members contain no embedded terms or notice. That route remains
invalid. The independent official model-repository route supplies the missing
scope: both exact model cards declare Apache-2.0 and their inference files are
byte-identical to all six BOS payload members. Current executable result:
`source_provenance_valid=true`, `hashes_valid=true`,
`official_model_repository_terms_proven=true`,
`artifact_level_redistribution_proven=true`, and
`conversion_permitted=true`.

## Blocked downstream gates

| Gate | Status | Direct reason |
|---|---|---|
| Artifact redistribution | PASS for conversion source | Exact official model-repository revisions scope Apache-2.0 to byte-identical BOS payloads |
| Notices | PASS for tracked source bundle; package binding BLOCKED | Canonical Apache-2.0 text SHA-256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` and reviewed model/change notice SHA-256 `8d81f5d0c58547cce471c24f82efe768a9d907d06764f67e90cc680c6d777729` are tracked; model-store and package checksums remain absent |
| Paddle-to-ONNX conversion provenance | PASS for conversion only | Exact source, toolchain, selected wheel hashes, installed RECORD files, command, logs, opset, and two byte-identical output hashes are bound |
| ONNX parity | PASS for conversion only | 16 deterministic CPU raw-tensor cases per model passed `1e-4` across four detector shapes and five recognizer widths; maxima were `1.349323213162279e-07` and `2.205371856689453e-06` |
| CPU execution | PASS for conversion only | Both checked ONNX files executed through CPU-only ONNX Runtime sessions |
| Frozen public/sealed OCR accuracy | FAIL | The once-only frozen 220-case evaluation produced validation and sealed exact match `0.0`, CER `1.0`, role accuracy `0.03` and `0.02`, and detection exact count `18/220` |
| Role accuracy | FAIL | Validation role accuracy was `0.03`; sealed-test role accuracy was `0.02` |
| Text masks and no marker creation | FAIL and BLOCKED | The detector produced 195 false regions across text cases and false regions on 7 of 20 exclusions; no independent composed marker-stage evidence was supplied |
| Model manifest | BLOCKED | Public and sealed benchmarks, production approval, model-store discovery, and packaging discovery are unresolved |

No private graph or Chandler data was read, tuned, or evaluated. Downloaded
archives, extracted payloads, and audit JSON remain ignored under
`ml/ocr/official_bakeoff/runs/`.

## Required next evidence

Do not rerun or tune against the exposed frozen split. The official pair needs
a newly preregistered upstream graph-text crop or masking contract and a new
sealed split before it can be reconsidered. Any future candidate must run exact
detection, recognition, role, exclusion, and independently composed
marker-creation gates without Chandler or private input. It must then bind the
reviewed Apache-2.0 license and ONNX change-notice bundle through both model
manifests, the production model store, and both package forms. Production
composition remains unavailable until both OCR tasks pass every model-store and
packaging requirement together.
