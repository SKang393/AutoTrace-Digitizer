<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

# GraphSR training toolchain notices

The GraphSR training, export, benchmark, and test toolchain uses unbundled
Python packages. The license or notice artifacts listed below are exact copies
from the installed distributions and are tied to the checksum-pinned wheel
records in `DEPENDENCY_PROVENANCE.csv`. Existing package notices were reused
where available.

| Dependency | License | Notice artifact | Reviewed wheel-entry SHA-256 |
|---|---|---|---|
| PyTorch 2.13.0 | BSD-3-Clause | `LICENSES/PyTorch-2.13.0-LICENSE.txt` | `0fc92425d130d9c87a5eb7427082e3afdbff6ef169bc678e15abba4c2d37c934` |
| ONNX 1.22.0 | Apache-2.0 | `LICENSES/ONNX-1.22.0-LICENSE.txt` | `3ddf9be5c28fe27dad143a5dc76eea25222ad1dd68934a047064e56ed2fa40c5` |
| ONNX 1.22.0 | NOTICE | `LICENSES/ONNX-1.22.0-NOTICE.txt` | `62c2c7bb3be2833f5e2c8a2576ac10666cf26aa09b7b631b55fed77bc7dc91c7` |
| ONNX Runtime 1.27.0 | MIT | `LICENSES/ONNXRuntime-1.27.0-LICENSE.txt` | `c250d6278f0b47a6439fb7592b08b58a55eb9f535aa49a1db63211c3f982b674` |
| NumPy 2.3.5 | BSD-3-Clause | `LICENSES/NumPy-2.3.5-LICENSE.txt` | `e5eb9d828cb6548a3fbcf66d8f6fbf71f0426f8e2de1d2c1601b620b9b2ea4a5` |
| Pillow 12.3.0 | MIT-CMU | `LICENSES/Pillow-12.3.0-LICENSE.txt` | `7cfc737a2d7fda776b8491589c69c5cd826e4b79c984dce51e8639211a83d961` |
| psutil 7.2.2 | BSD-3-Clause | `LICENSES/psutil-7.2.2-LICENSE.txt` | `b89c063b3786e28e0c0a38f1931db61fed35e69dd2a2966fbecffee0f46c8d10` |
| pytest 9.1.1 | MIT | `LICENSES/pytest-9.1.1-LICENSE.txt` | `0cc5ea2ce3cbae09ce9581e64a70bff7b74fef07f6b295597ab6a2cdead840d1` |
| jsonschema 4.26.0 | MIT | `LICENSES/jsonschema-4.26.0-COPYING.txt` | `66010c33da1256c30c4359f6edd8c30fbccbf41a4bf66cb0c576e5c726978142` |

These dependencies, their wheels, and their native libraries are not bundled
in the Windows application by Session 07. The copied notice texts are public,
contain no study data, and are Git eligible. Downloaded wheels and local Python
environments remain outside Git.

The GraphSR candidate uses no pretrained third-party weights. A future
qualifying model artifact is original Graph Auto Reader work under Apache-2.0,
subject to a separate exact-artifact and release-storage review.
