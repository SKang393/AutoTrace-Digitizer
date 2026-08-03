# Marker classifier experiment comparison

## Fixed protocol

- Deterministic seed: `20260803`
- Experiment budget: 3
- Training epochs: 28
- Selection data: `vector_thin` and `press_heavy` training families, then
  `scan_soft` validation family
- Held-out data: sealed until the selected checkpoint and ONNX are fixed
- Local gate: shape macro-F1 and fill macro-F1 each at least 0.90
- Gate authority: session-local preregistration, not maintainer agreement

## Experiments

| ID | Architecture | Learning rate | Validation result | Decision |
|---|---|---:|---|---|
| E1 | compact depthwise encoder, global pooling, four separate heads | 0.003 | Shape macro-F1 0.1631; fill macro-F1 0.6105 | Rejected: global pooling discarded spatial shape geometry |
| E2 | compact depthwise encoder, 4x4 spatial projection, four separate heads | 0.003 | Shape macro-F1 0.5016; fill macro-F1 0.6830 | Rejected: translation-sensitive projection did not generalize |
| E3 | compact spatial CNN, deterministic translation augmentation, four separate heads | 0.003 | Shape macro-F1 0.8711; fill macro-F1 0.9815 | Selected and frozen after budget exhaustion |

The experiment budget is exhausted. No further model or threshold changes may
be made in response to the sealed held-out result.

## Single sealed held-out result

E3 was frozen and exported before the held-out split was opened. The one sealed
evaluation passed the session-local gates: shape macro-F1 `0.9251`, fill
macro-F1 `0.9661`, artifact F1 `1.0000`, embedding top-1 retrieval accuracy
`0.9352`, and minority star/asterisk/cross macro-F1 `1.0000`. ONNX CPU parity
maximum absolute error was `9.5367431640625e-06`, within the preregistered
`1e-5` tolerance. No held-out rerun or post-test tuning was performed.
