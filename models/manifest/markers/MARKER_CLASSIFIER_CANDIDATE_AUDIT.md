<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

# Goal 19 marker model candidate audit

Audit date: 2026-08-03

## Decision

The marker classifier is a reproducible, unapproved candidate. The marker
center model remains experimental and failed. Neither model is approved for a
production or release composition.

Generated datasets, checkpoints, ONNX files, and command reports remain ignored
under `ml/markers/*/artifacts/goal19-reproduction`. No private or external data
was read or used.

## Fixed experiment protocol

- Seed: `20260803`.
- Environment: Python `3.13.7`, PyTorch `2.13.0+cpu`, NumPy `2.3.5`, ONNX
  `1.22.0`, ONNX Runtime `1.27.0`, and Pillow `12.3.0`.
- Host: Windows `10.0.26200`, AMD Ryzen 7 5800X, 8 physical and 16 logical
  cores.
- Center split: tracked procedural train and validation families only.
- Classifier split: tracked procedural train and validation families only.
- Held-out evaluations in Goal 19: `0`.
- Experiment budget: one byte-exact reproduction of each already-selected
  architecture. No parameter change or threshold sweep was added.

The earlier sealed test metrics are already public in tracked Session 10 and 11
evidence. Reopening those same splits would not create an independent held-out
result, so this audit did not run either benchmark command.

## Commands

```powershell
python -m pytest ml/markers/center/tests ml/markers/classifier/tests -q
python -m ml.markers.center.train --output ml/markers/center/artifacts/goal19-reproduction
python -m ml.markers.classifier.train --output ml/markers/classifier/artifacts/goal19-reproduction
python -m ml.markers.center.export --checkpoint ml/markers/center/artifacts/goal19-reproduction/marker-center.pt --output ml/markers/center/artifacts/goal19-reproduction/marker-center.onnx --report ml/markers/center/artifacts/goal19-reproduction/onnx-parity.json
python -m ml.markers.classifier.export --checkpoint ml/markers/classifier/artifacts/goal19-reproduction/marker-classifier.pt --output ml/markers/classifier/artifacts/goal19-reproduction/marker-classifier-packed.onnx --report ml/markers/classifier/artifacts/goal19-reproduction/onnx-parity.json
$env:GRAPHREADER_MARKER_CLASSIFIER_CANDIDATE = (Resolve-Path 'ml/markers/classifier/artifacts/goal19-reproduction/marker-classifier-packed.onnx').Path
dotnet test tests/GraphReader.Markers.Tests/GraphReader.Markers.Tests.csproj -c Release --filter "FullyQualifiedName~CandidateClassifierProviderIntegrationTests" --logger "console;verbosity=detailed"
```

## Marker center reproduction

- Architecture: `compact-pplcnet-depthwise-fpn-v1`.
- Training time: `25281.500` ms.
- Validation standard-mask F1 at 5 px: `1.0000`.
- Validation zero-mask F1 at 5 px: `1.0000`.
- Validation duplicates: `0`.
- Validation hard-negative hits: `0` for all eight kinds.
- Checkpoint SHA-256:
  `72471156daec005b7617916a3d81c3966c0a5e36cb93c6c88aaa58f4d1272fb9`.
- ONNX SHA-256:
  `061a496167382d1bd11bb580bed383d2d1725da2001f9c440b7f1acc59ac116a`.
- CPU ONNX parity maximum absolute difference:
  `2.384185791015625e-06`, below `1e-05`.

These are byte-identical to the prior checkpoint and ONNX. The model remains
failed because its preserved historical held-out result had exact counts in
only 5 of 6 fixtures, and that report predates the production raw-mask gate.

## Marker classifier reproduction

- Architecture: `compact-spatial-cnn-patch-classifier-v3`.
- Training time: `22057.727` ms.
- Validation shape macro-F1: `0.8710622710622711`, below local gate `0.90`.
- Validation fill macro-F1: `0.9814671814671815`.
- Validation artifact F1: `1.0000`.
- Validation embedding top-1 retrieval: `0.9722222222222222`.
- Checkpoint SHA-256:
  `9f15d0d2ef067418b22ca625e405775c771dd63ea798f7c63fbed43d8d50b393`.
- Packed ONNX SHA-256:
  `59b4af98fe40abd436f01a8c14bf0d12a7c82682ec072c65cef92881aa18b0ef`.
- CPU packed parity maximum absolute difference:
  `3.814697265625e-06`, below `1e-05`.

The exact checkpoint previously passed one sealed procedural held-out run:
shape macro-F1 `0.925106`, fill macro-F1 `0.966116`, artifact F1 `1.0`, and
minority-shape macro-F1 `1.0`. That historical benchmark used the separate-head
ONNX. It was not rerun for the packed runtime wrapper.

The exact packed ONNX then passed a real .NET ONNX Runtime probe:

- CPU inference: `7.1318` ms.
- DirectML inference: `253.1578` ms.
- Maximum CPU versus DirectML difference:
  `1.9073486328125e-06`, below `1e-04`.
- Output: 50 floats for input shape `[2,1,32,32]`, matching `[2,25]`.

## Exact ignored evidence

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Center checkpoint | 59,443 | `72471156daec005b7617916a3d81c3966c0a5e36cb93c6c88aaa58f4d1272fb9` |
| Center ONNX | 37,542 | `061a496167382d1bd11bb580bed383d2d1725da2001f9c440b7f1acc59ac116a` |
| Center training report | 5,938 | `67546514df20d8b3b856f1fda8d5caa923fff5dea985adf833ce499e98e4a7b4` |
| Center parity report | 1,573 | `1155600b86075f367e4de8dfd5258541e4d97f7f97a7517b5d364dee468f37e5` |
| Classifier checkpoint | 322,333 | `9f15d0d2ef067418b22ca625e405775c771dd63ea798f7c63fbed43d8d50b393` |
| Packed classifier ONNX | 320,448 | `59b4af98fe40abd436f01a8c14bf0d12a7c82682ec072c65cef92881aa18b0ef` |
| Classifier training report | 4,196 | `e7df6aca9c3eff6ec5d3b5580cf70194eaaeeeeff7f4054c5f3c9427ceef8cc1` |
| Classifier parity report | 2,692 | `2d3bf8dc76b1076b726510d95799ba59af03d2508dfc6ffc844a23cbfca36e48` |

## License and release blockers

Project training code, procedural data, checkpoints, and exports are original
Apache-2.0 work. Training dependencies remain unbundled and are covered by the
tracked dependency ledger and notices.

Classifier release blockers:

1. Validation shape macro-F1 misses the session-local gate.
2. The exact packed wrapper has no direct sealed held-out benchmark.
3. The local `0.90` gates are not maintainer-approved acceptance thresholds.
4. Private graph validation has not passed.
5. Identical installer and portable model discovery, checksum, offline CPU
   fallback, and notice inclusion have not passed.

Center release blockers remain those in `PROVENANCE_BLOCKER_AUDIT.md`.
