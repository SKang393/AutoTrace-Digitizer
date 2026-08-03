# Selected marker classifier evidence

## Source and data

- Training source bundle SHA-256: `6e9c0ff661d224e38363ad5cad6775e8b088623f6961902ee4a1cf2ad61dcd85`
- Packed runtime export source exact-file SHA-256 for `export.py`:
  `ba985ef5488cd8e927a3982854feb319ea083a782e7fae2b09a8386d933ccfb9`
- Bundle algorithm: canonical UTF-8 LF bytes for sorted `__init__.py`,
  `dataset.py`, `metrics.py`, `model.py`, and `train.py`; material lines are
  `<relative-name>=<file-sha256>\n`, then SHA-256 of their concatenation.
- Selection manifest canonical UTF-8 LF JSON SHA-256:
  `4832b281df860c9d433406346e397303b307ce74efcee75fbe09d00e969aa44f`
- The local Windows manifest file has CRLF byte SHA-256
  `90b41c4280d617ee54f92e82d87f8b1faf50e4e078ffa174206f3203a1ee3563`.
  The checkpoint stores the canonical content digest above.
- Held-out full-manifest canonical UTF-8 LF JSON SHA-256:
  `ad568f763de1a3701b2d523321321e3ee4572b996fe1dda9a5627ed8637a3857`
- Inputs are original procedural Apache-2.0 project data. No external,
  private, article-derived, or pretrained inputs were used.

## Selected artifacts

Generated artifacts remain ignored and local under
`artifacts/session11-final-e3/` and `artifacts/session11-runtime-packed/`.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `marker-classifier.pt` | 322333 | `9f15d0d2ef067418b22ca625e405775c771dd63ea798f7c63fbed43d8d50b393` |
| `marker-classifier.onnx` | 318787 | `03aabaf8b4793bf734f60164b2c98f43ce60050973fdda957975c4cf5b711a81` |
| `marker-classifier-packed.onnx` | 320448 | `59b4af98fe40abd436f01a8c14bf0d12a7c82682ec072c65cef92881aa18b0ef` |
| `training-report.json` | 4193 | `a0eb57548f652813bc5b1845f1eb054f895e331157f71a5d4c0c23d58717c018` |
| `onnx-parity.json` | 1417 | `abb580cf1062a5d7dba05115ba9fef42d214320f433fadcc657c13ee9d7ffa7c` |
| `benchmark.json` | 7521 | `b93e33b0c901870aa7d8b141dc936fd28dbf0a2727fee300e4da50772fa78382` |
| `heldout-evaluation.seal.json` | 136 | `dcede728b64f07edb4cf1b7b73c08765edae78e7a2481ef1afb1eb78c03b13d2` |
| packed `onnx-parity.json` | 2691 | `1df507b7961458f44131da978139f1bc799af9e12fed6a23c68f6a44a1d7b68e` |

The first ONNX artifact exposes four separate training outputs and is retained
only as historical evidence for the already sealed checkpoint evaluation. It
is not the C# runtime artifact. The packed ONNX is the integration artifact. It
  exposes one output named `classification_heads`, shaped `[N,25]`, ordered as
  nine temperature-scaled shape logits, three temperature-scaled fill logits,
  one artifact logit, and twelve embedding values. It reads the validation-fit
  temperatures from the unchanged selected checkpoint, so C# softmax confidence
  corresponds to the reported calibration evidence. No retraining is performed.

## Results

- Selection experiment count: 3 of 3; all selection used validation only.
- Selected training time: 18,989.940 ms on CPU.
- Validation: shape macro-F1 0.871062, fill macro-F1 0.981467,
  artifact F1 1.0, embedding top-1 retrieval accuracy 0.972222.
- Single sealed held-out evaluation: shape macro-F1 0.925106, fill macro-F1
  0.966116, artifact F1 1.0, embedding top-1 retrieval accuracy 0.935185,
  and minority star/asterisk/cross macro-F1 1.0.
- Confidence calibration: shape ECE 0.040946 and fill ECE 0.032179.
- Held-out ONNX Runtime CPU inference: 176.847 ms total for 420 patches,
  0.421065 ms per patch.
- Held-out PyTorch/ONNX maximum absolute error: `9.5367431640625e-06`,
  within the preregistered `1e-5` tolerance.
- Validation-only packed runtime ONNX maximum absolute error:
  `3.814697265625e-06`. Every PyTorch packed slice matched its separate
  runtime transform exactly before ONNX comparison. The checkpoint temperatures
  were shape `1.35` and fill `0.7`. No held-out data was opened during this
  integration export.

The held-out metrics above are evidence for the unchanged checkpoint. They are
not presented as a new benchmark of the packed runtime wrapper, and the sealed
held-out evaluation was not rerun.

The local technical model gates pass. Release eligibility remains false until
the separately owned model manifest, C# runtime provider, packaging audit, and
maintainer-approved acceptance threshold are complete.
