<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

# Goal 19 graph-numeric CTC experiment audit

Audit date: 2026-08-03

## Decision

The project-generated graph-numeric CTC model is **failed experimental work**,
not a candidate and not approved. No model manifest was created, no automatic
OCR stage was enabled, and no weight is eligible for Git or release packaging.

The experiment proved that the tracked toolchain can emit a valid ONNX with the
runtime tensor contract and execute it through CPU and DirectML. It did not
meet the fixed validation or held-out quality gates.

## Fixed data and experiment budget

- Seed: `20260803`.
- Corpus: 768 train, 192 validation, and 192 held-out test labels.
- Corpus manifest SHA-256:
  `b50fb1ee5ed457a82b3755c0588851898f71628d553710164ba02e55a6c179e8`.
- Split boundary: renderer, vector glyph, and degradation families are
  disjoint.
- Glyph source: project-owned 5 by 7 vector definitions in
  `ml/ocr/synthetic.py`.
- No Pillow font, Windows font, external font binary, pretrained weight,
  downloaded model, private graph, article image, or external dataset was
  read or used.
- Candidate A: 24 epochs at learning rate `0.003`.
- One preregistered targeted repair: 96 epochs, with all other inputs,
  parameters, code, preprocessing, and gates unchanged.
- No further run, threshold sweep, or test-set evaluation is permitted for
  this defect class under Goal 19.

Session-local gates were validation and test exact match at least `0.90`, test
character error rate at most `0.05`, ONNX parity at most `1e-4`, and exact
runtime output shape `[N,32,14]`. These were never maintainer-approved release
gates.

## Results

| Measure | Candidate A, 24 epochs | Repair, 96 epochs | Gate |
|---|---:|---:|---:|
| Validation exact match | 0.03125 | 0.03125 | >= 0.90 |
| Held-out exact match | 0.015625 | 0.015625 | >= 0.90 |
| Held-out character error rate | 0.8560747663551402 | 0.9327102803738317 | <= 0.05 |
| PyTorch versus CPU ONNX maximum difference | 7.152557373046875e-07 | 5.7220458984375e-06 | <= 1e-4 |
| Training time, ms | 44086.796100018546 | 165551.87790002674 | reported |

The repair reduced training loss from `2.1881163517634072` at epoch 24 to
`0.26883506029844284` at epoch 96 but did not generalize across the held-out
vector glyph and degradation families. This is a model/data-generalization
blocker, not a packaging or threshold issue.

## Ignored local artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Candidate A report | 2,891 | `c883442e770720d026afa7de431b6adc75377aee13190467d6c899774234526a` |
| Candidate A checkpoint | 91,813 | `47c96d79ca963855734daa9ed28607a4c37ef5c84656ebafbcf2226209eee7ad` |
| Candidate A ONNX | 90,220 | `734e27b85ffb95c56951216f2c0c7c6902eb385ddb7fff3a41a1eac9c020368b` |
| Repair report | 4,698 | `94d4056091f9a5aff3cfbae49ed336a55cfc331912290ebbbfb4f3fc314c412d` |
| Repair checkpoint | 91,813 | `7df0485ac28db69ec7924f4a03801921cc3bf9eb62ed7cff2244d3fc67a8931b` |
| Repair ONNX | 90,220 | `a48d640226fd95aa67316837abd5a8d08258320b042a5b6a24ea32ee1ab6aa91` |

The artifacts are under ignored `ml/ocr/runs/`. They contain only generated
model state and reports. They may be deleted and reproduced, but may not be
committed, bundled, or described as a candidate.

## Runtime provider evidence

The exact repair ONNX passed the opt-in .NET provider probe using the production
ONNX Runtime adapter:

- Providers reported: `DmlExecutionProvider`, `CPUExecutionProvider`.
- Input: `[2,1,32,128]`.
- Output: 896 floats, matching `[2,32,14]`.
- CPU inference: `4.6307` ms.
- DirectML inference: `247.0504` ms.
- CPU versus DirectML maximum absolute difference:
  `1.9073486328125e-05`, below `1e-4`.

This proves execution compatibility only. It does not repair the failed OCR
quality metrics or establish exclusive GPU node assignment.

## Source, license, privacy, and Git eligibility

- Source: original Graph Auto Reader training code and procedural vector glyphs.
- Purpose: graph-specific numeric OCR model research and runtime-contract proof.
- Model copyright and license if a future checkpoint passes: Copyright 2026
  Sungwoo Kang, Apache-2.0.
- Training tools: permissive, unbundled local dependencies recorded in
  `ml/ocr/DEPENDENCY_PROVENANCE.csv`.
- Privacy: public procedural data only; no private or supplied-example data.
- Git eligibility: training source, tests, plan, and this audit are eligible.
  Generated samples, checkpoints, ONNX files, and reports are ignored and not
  eligible.

## Remaining gates

1. Design a new preregistered experiment outside this exhausted defect budget
   that generalizes across held-out glyph and degradation families.
2. Meet a maintainer-approved numeric OCR threshold.
3. Add a schema-valid manifest and model notice only after a model is a truthful
   candidate.
4. Validate the exact accepted model on private graphs without training on
   them.
5. Validate packaged discovery, checksum, offline CPU fallback, DirectML,
   notices, installer, and portable parity.
6. Separately resolve text-region detection and general text recognition; this
   experiment addresses only graph-numeric recognition.
