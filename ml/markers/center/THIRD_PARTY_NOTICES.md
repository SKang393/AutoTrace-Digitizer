# Marker-center training toolchain notices

The training and export toolchain uses unbundled Python packages. The exact
license or notice bytes below were extracted from the checksum-pinned wheels
listed in `DEPENDENCY_PROVENANCE.csv`.

| Dependency | License | Exact package artifact | SHA-256 |
|---|---|---|---|
| PyTorch 2.13.0 | BSD-3-Clause | `LICENSES/PyTorch-2.13.0-LICENSE.txt` | `0fc92425d130d9c87a5eb7427082e3afdbff6ef169bc678e15abba4c2d37c934` |
| ONNX 1.22.0 | Apache-2.0 | `LICENSES/ONNX-1.22.0-LICENSE.txt` | `3ddf9be5c28fe27dad143a5dc76eea25222ad1dd68934a047064e56ed2fa40c5` |
| ONNX 1.22.0 | NOTICE | `LICENSES/ONNX-1.22.0-NOTICE.txt` | `62c2c7bb3be2833f5e2c8a2576ac10666cf26aa09b7b631b55fed77bc7dc91c7` |
| ONNX Runtime 1.27.0 | MIT | `LICENSES/ONNXRuntime-1.27.0-LICENSE.txt` | `c250d6278f0b47a6439fb7592b08b58a55eb9f535aa49a1db63211c3f982b674` |
| NumPy 2.3.5 | BSD-3-Clause | `LICENSES/NumPy-2.3.5-LICENSE.txt` | `596b953e1dcbe829b32ae444387efa15003fc6abdeca8a2179f364a6364c286e` |
| Pillow 12.3.0 | MIT-CMU | `LICENSES/Pillow-12.3.0-LICENSE.txt` | `4f7866a74802c6326f81faff59a56546b6aec2b10b91973e0e9308de95e79857` |
| pytest 9.1.1 | MIT | `LICENSES/pytest-9.1.1-LICENSE.txt` | `ca836a5f9ecca3b2f350230faa20a48fb8b145653b5568d784862df864706b9b` |

These dependencies, their wheels, and their native libraries are not included
in the Windows application distribution by this training session. The license
artifacts are public, contain no private data, and are Git eligible. The ignored
wheel audit cache is not Git eligible. The trained model uses only original
procedural project data and original Apache-2.0 project code; no third-party
weights are used.
