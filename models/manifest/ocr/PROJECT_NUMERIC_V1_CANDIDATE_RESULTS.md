<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Project numeric OCR V1 candidate results

Audit date: 2026-08-04

## Decision

The fixed three-candidate budget is exhausted and no project numeric OCR model
is approved. All candidates failed the `0.90` validation exact-match gate, so no
Candidate 2 or Candidate 3 sealed metric was opened and neither produced ONNX.
No fourth candidate, rerun, recovery, parameter change, or threshold sweep is
authorized for this defect class.

## Direct results

| Candidate | Single change | Validation exact | Validation CER | Role accuracy | Exclusion | Sealed status | ONNX |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | Frozen baseline | `0.3359375` | `0.537696335078534` | `0.5828125` | `1.0` | Legacy recovery evaluated and failed | None |
| 2 | Renderer/degradation only | `0.34375` | `0.37801047120418846` | `0.821875` | `1.0` | Unopened | None |
| 3 | Architecture only | `0.408203125` | `0.2973821989528796` | `0.98125` | `1.0` | Unopened | None |

Candidate 2 ran once from commit
`cade7a16d9044b11f8c402ac49206464cd0c5b03` in `114661.40760001144` ms.
Its ignored report is 8,717 bytes with SHA-256
`5a29eb78d29e5ada811da12cd27e6456ccb3bb172766aec8a6a20da3c0b5cfa6`.
Its ignored checkpoint is 2,150,605 bytes with SHA-256
`5ba4da69d0a3b229ef55dae29e1d64d4b61ebfbe361114a027189f50ecbe0b45`.

Candidate 3 ran once from commit
`4083738143df7b7dc12946cd6280c6ddeebd16c9` in `381724.58449999976` ms.
Its ignored report is 8,640 bytes with SHA-256
`7997df069ceff0599d27434abc85918bc5aa223fe8e7f9d77f61f1ace4b36b4d`.
Its ignored checkpoint is 2,170,611 bytes with SHA-256
`21f331b77200156adc0c45246145d7e3ebe3066aa152819b2eb6f82df5f8a2ea`.

Both later candidates retained the sealed fingerprint
`a4e2e8c0623d77a52d88da9b997deebe9bb57245a65e25c918b6817892b39aee`
with `metrics_opened=false` and `predictions_opened=false` because validation
failed first.

## Provenance and license

Training used only deterministic project-owned procedural graph labels and
exclusion shapes. No Chandler image, private graph, article data, external
dataset, font, downloaded weight, or pretrained weight was used. Source and
generated project weights are Apache-2.0, but failed checkpoints remain ignored,
unapproved, non-package-eligible, and not Git eligible.

## Production consequence

The project-trained branch is closed without a passing candidate. The official
PP-OCRv5 branch also remains blocked before conversion because the exact
archives contain no artifact-scoped license, commercial-use grant,
redistribution grant, or notice. Production Auto Detect must continue to report
OCR unavailable. The release audit must continue to fail closed and emit no
public artifact.
