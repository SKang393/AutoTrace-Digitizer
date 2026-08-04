<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

# Graph-numeric canonical-slot V3 experiment audit

Audit date: 2026-08-04

## Decision

Sequence V3 is **failed historical research**. Candidate A consumed the first
and only sealed test observation and failed. Candidate B and C reused those
records, so their results are nonsealed observations and cannot support model
selection or promotion. No manifest was created, and no generated weight is
approved, bundled, or release eligible.

## Protocol and data

- Protocol ID: `graph-numeric-sequence-v3-20260804`.
- Seed `20260804`; 2,048 train, 512 validation, 512 designated test.
- 24 epochs, Adam `0.002`, batch size 64, maximum candidates A/B/C.
- Corpus manifest SHA-256:
  `18b715012a1626a85c4c42832a5d28b1b584d516dd1801fdff38ce1d0dc9e35a`.
- Procedural project data only. No Chandler, private graph, article data,
  external dataset, external font, or pretrained weight was used.

The renderer/font/degradation labels differ, but their implementations are not
independent. They share `_GLYPHS` and `_render`; Candidate C inverts the shared
generator's duplicate/delete column transformations. These results provide no
independent-renderer or production-generalization evidence.

## Corrected result classification

| Measure | Candidate A | Candidate B | Candidate C |
|---|---:|---:|---:|
| Validation exact | 0.22265625 | 0.1640625 | 1.0 |
| Holdout exact | 0.828125, first sealed | 0.810546875, reused nonsealed | 0.890625, reused nonsealed |
| Holdout CER | 0.051551814834297736 | 0.0583903208837454 | 0.03156233561283535 |
| Historical all-zero export-smoke parity | 7.62939453125e-06 | 1.52587890625e-05 | 1.52587890625e-05 |
| Training time, ms | 204927.43759999576 | 261254.2753000016 | 304634.7378999999 |

The historical parity values above were measured on an all-zero export tensor,
not representative samples. The ignored `representative-parity.json` records a
fresh comparison of the exact existing Candidate C artifacts on all 512
validation and all 512 reused holdout inputs, plus dynamic batches of 1, 2,
and 17. The actual representative maximum was `3.0517578125e-05` on CPU, below
`1e-4`; all 1,024 decoded predictions matched. This check does not train a
candidate and does not restore sealed evidence.

## Artifact identity

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Candidate A report | 3,093 | `b968b5e64400c217c1ae1366d18e208823984190bb85706a4329b014d5d45400` |
| Candidate A checkpoint | 48,393 | `95e1abd6cb7473590b6917ebc88672321d195fa0517c130f83c4880c380f4b0c` |
| Candidate A ONNX | 53,096 | `1af2477d111d6ae93d479351617ba4bd5c8180dc552dbaad788d0cea11effc92` |
| Candidate B report | 3,094 | `b4f255a0abbd7d4fa1867429280e73ce351a4ed6581e2c083e0d9a9c19ba4fb8` |
| Candidate B checkpoint | 48,393 | `dcd9f00389f9bdaa2513d81fd0adfedc91576873855efb6e2c06f0cb4d82126c` |
| Candidate B ONNX | 53,096 | `dfe7a978789d36f71f02b0bbdae07a17d2d6551efa1a0bd7d369fe22184d7e20` |
| Candidate C report | 3,065 | `a686f9c322c58d29351a4e58e27fa198265748dd00eeee2e080a564ef93afd52` |
| Candidate C checkpoint | 48,393 | `dcd9f00389f9bdaa2513d81fd0adfedc91576873855efb6e2c06f0cb4d82126c` |
| Candidate C ONNX | 53,096 | `dfe7a978789d36f71f02b0bbdae07a17d2d6551efa1a0bd7d369fe22184d7e20` |

Generated artifacts remain ignored under `ml/ocr/sequence_v3/runs/`.

## Source and protocol binding

`ml/ocr/sequence_v3/SOURCE_BINDING.json` binds the current tracked evaluator,
shared generator, preprocessor, model, protocol, metrics, and this plan to exact
SHA-256 values. It also binds and verifies Candidate A/B/C reports, Candidate C
checkpoint and ONNX, and the representative parity report by exact paths, byte
counts, and SHA-256. Controlled tests prove mutation and missing ignored
artifacts fail. Historical A and B source snapshots were not preserved, so
their reports cannot be retroactively source-bound to source. That limitation
is retained as a blocker.

## PP-OCRv5 audit reconciliation

The two official PP-OCRv5 archives were downloaded solely into ignored storage
for archive hashing, member inventory, and artifact-terms inspection. They were
not converted or benchmarked. Exact artifact redistribution evidence is absent,
so the official path remains blocked as documented in
`PP_OCRV5_OFFICIAL_ARCHIVE_AUDIT.md`.

## Remaining blockers

- No valid sealed result exists after Candidate A failed.
- Historical A/B source snapshots are unavailable.
- Shared-generator inversion invalidates family-independence confidence.
- Production preprocessing, private validation, DirectML, packaging, notices,
  installer/portable parity, and the release audit remain unresolved.
