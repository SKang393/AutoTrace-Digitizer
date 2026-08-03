<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

# GraphSR-x2 candidate notice

## Candidate identity

- Model ID: `graphsr-x2-candidate`
- Version: `0.1.0-local-candidate`
- Local candidate artifact: `graphsr-x2.onnx`, 207,110 bytes
- ONNX SHA-256: `4b0237683cd61ecd639015380bad9323a5fe79b295ffebf0c93720f51ef0d667`
- Checkpoint SHA-256: `9524033b57b37b78e4f35bcc6256412bf087ec1cace4c5f857eb1eb51632847a`
- Dataset identity: `sha256:74a5d03366808d8c1ae0b41439d0f6e613490a400296c8de21c0b85c0e39bfc6`
- ONNX opset: 18
- Artifact status: deterministic 100-step public-synthetic candidate, rejected, ignored, and unbundled
- Default status: not selected
- Release status: not eligible

The manifest records the exact ignored local ONNX bytes. ONNX checker and CPU
parity passed on one even and one odd dynamic input shape. Maximum absolute
error was `2.384185791015625e-07` against tolerance `1e-05`.

Attempt 2 used 100 optimizer steps over 16 deterministic two-stage degraded
public-synthetic crops with a corrected per-marker loss. The fixed two-case
held-out benchmark then rejected the artifact for marker-center, thin-line,
and open-marker fidelity. It is not a production model and may not enter
release staging.

## Copyright and license

Copyright 2026 Sungwoo Kang.

The original Graph Auto Reader GraphSR source, procedural training data, and
any qualifying original trained artifact are licensed under Apache License
2.0. The complete license text is in the repository root `LICENSE` file. This
notice does not license third-party pretrained weights, and none are used by
the candidate.

Commercial use of original Apache-2.0 work is allowed. Redistribution of this
GraphSR candidate is currently recorded as false because the held-out fidelity
gate failed and provider coverage, tracked-revision binding, and release-storage
review remain incomplete.

## Data and privacy

Only deterministic local procedural project scenes are eligible for training.
No private studies, published figures, participant data, human annotations,
downloaded weights, or network services are included.

## Required release amendment

Before any model distribution, bind the model to an exact source commit and
sealed dataset manifest, train a replacement candidate without reopening this
held-out split for tuning, pass a separately frozen benchmark, verify required
providers, and record the reviewed storage location. Installer and portable
artifacts must contain identical verified bytes and notices.
