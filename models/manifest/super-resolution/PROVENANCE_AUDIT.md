<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

# Real-ESRGAN super-resolution provenance audit

Audit date: 2026-08-03

## Scope and decision

This directory contains metadata only. No weights, converted models, runtime
binaries, test images, or benchmark outputs are included.

The three required benchmark identities have official upstream artifacts, but
they do not share one runtime format:

- `RealESRGAN_x2plus` is recorded as the official PyTorch `.pth` reference
  weight. It is not directly runnable by `realesrgan-ncnn-vulkan`.
- `realesr-general-x4v3` is recorded as the official PyTorch `.pth` reference
  weight with final output scale 2. It is not directly runnable by
  `realesrgan-ncnn-vulkan`.
- `realesr-animevideov3` is recorded from the official NCNN Windows package,
  using its scale-2 `.param` and `.bin` pair with the Vulkan provider.

The official releases inspected here do not contain checksum-verifiable NCNN
payloads for `RealESRGAN_x2plus` or `realesr-general-x4v3`. Those two models
must remain benchmark references until an official NCNN artifact or an
explicitly approved conversion and provenance workflow exists.

## Required local audit fields

| Dependency/model | Version | Source | License | Bundled or downloaded | Notice path | Checksum | Review status |
|---|---|---|---|---|---|---|---|
| `RealESRGAN_x2plus` | `v0.2.1` | Official Real-ESRGAN release | BSD-3-Clause | Not bundled; downloaded to ignored audit storage only | `LICENSES/Real-ESRGAN-BSD-3-Clause.txt` | `49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb` | Artifact verified again on 2026-08-03; current NCNN adapter is incompatible; production approval blocked |
| `realesr-general-x4v3` | `v0.2.5.0` | Official Real-ESRGAN release | BSD-3-Clause | Not bundled; downloaded to temporary audit storage only | `LICENSES/Real-ESRGAN-BSD-3-Clause.txt` | `8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292` | License and checksum reviewed; benchmark not run; NCNN integration blocked |
| `realesr-animevideov3` NCNN x2 | `v0.2.5.0-ncnn-x2` | Official Real-ESRGAN Windows NCNN package | BSD-3-Clause model notice | Not bundled; downloaded to ignored audit storage only | `LICENSES/Real-ESRGAN-BSD-3-Clause.txt` | Package: `abc02804e17982a3be33675e4d471e91ea374e65b70167abc09e31acb412802d` | Exact 2x adapter/runtime smoke passed on two public synthetic cases; required accuracy, memory, CPU fallback, and runtime notice gates remain blocked |

## Pinned official sources and verified hashes

### RealESRGAN_x2plus

- Revision: `v0.2.1@64ad194ddaf9c4d8c4b0d1b98cac6d89d3ea0d11`
- URL: <https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth>
- Size: `67,061,725` bytes
- SHA-256: `49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb`
- Format/provider: PyTorch checkpoint; CPU or CUDA through the official Python
  implementation.

### realesr-general-x4v3

- Revision: `v0.2.5.0@685d429c81888252bdb10f56c7754baededc3823`
- URL: <https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth>
- Size: `4,885,111` bytes
- SHA-256: `8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292`
- Format/provider: PyTorch checkpoint; CPU or CUDA through the official Python
  implementation.
- Output scale 2: the native x4 result is resized to exact x2 with OpenCV
  Lanczos4 by the official `outscale` postprocessing path.

### realesr-animevideov3 NCNN x2

- Model revision: `v0.2.5.0@685d429c81888252bdb10f56c7754baededc3823`
- Runtime revision: `v0.2.0@37026f49824c5cf84062e7c6a5dd71445dcf610f`
- Package URL: <https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip>
- Package size: `45,474,481` bytes
- Package SHA-256: `abc02804e17982a3be33675e4d471e91ea374e65b70167abc09e31acb412802d`
- `models/realesr-animevideov3-x2.param`: `3,173` bytes;
  SHA-256 `b88ff4f00ebf019a7fdac17fdd45a7fd3665d37509efc5baf2e4da2e24420a04`
- `models/realesr-animevideov3-x2.bin`: `1,247,368` bytes;
  SHA-256 `548a36f9c3f4ab8da56cd3b13badf23968bee207b396dad14d04b830e5f2ab2d`
- Format/provider: NCNN parameter and weight pair; Vulkan.

For comparison, the official Python reference weight
`realesr-animevideov3.pth` from the same v0.2.5.0 release is `2,504,012`
bytes with SHA-256
`b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d`.
It is not the file declared by the NCNN manifest.

## NCNN package boundary

The separately published official
`realesrgan-ncnn-vulkan-v0.2.0-windows.zip` has SHA-256
`1bbbdb12d470af80b035c773682e144c6c2f6ece9210832a289af0a48ce3fa9a`
and contains the executable and supporting files but no model payloads. The
larger Real-ESRGAN v0.2.5.0 Windows package contains only these model families:

- `realesr-animevideov3-x2`, `-x3`, and `-x4`;
- `realesrgan-x4plus`;
- `realesrgan-x4plus-anime`.

It does not contain `RealESRGAN_x2plus` or `realesr-general-x4v3` NCNN files.
The NCNN runtime package itself also requires its full MIT and transitive
notices before any release bundling. This model audit does not authorize or
bundle that runtime.

## License and redistribution review

The pinned official Real-ESRGAN revisions publish the project and official
release weights under BSD-3-Clause and publish no separate model-only
restriction for these assets. The existing local notice at
`LICENSES/Real-ESRGAN-BSD-3-Clause.txt` is substantively identical to the
upstream license after whitespace normalization.

BSD-3-Clause permits commercial use and redistribution when its notice,
conditions, disclaimer, and non-endorsement condition are preserved. The
manifest flags therefore record `commercial_use: true` and
`redistribution: true`; release packaging must retain the notice.

The NCNN application source is MIT-licensed, but its executable package has a
separate runtime/transitive-notice review boundary. Model BSD metadata must not
be treated as a complete runtime license audit.

## Privacy and Git eligibility

All audited artifacts are public official upstream releases and contain no
project or article data. Manifest JSON and this audit are Git-eligible.
Weights, NCNN model payloads, runtime binaries, downloaded archives, extracted
files, and benchmark images remain excluded.

## Goal 19 local acquisition and adapter evidence

Before download, the local audit record captured source, purpose, expected
checksum, license, privacy status, and Git eligibility. Both authorized assets
were downloaded only below the ignored directory
`artifacts/goal19-realesrgan/`.

Measured artifact evidence:

- `RealESRGAN_x2plus.pth`: `67,061,725` bytes, SHA-256
  `49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb`;
- `realesrgan-ncnn-vulkan-20220424-windows.zip`: `45,474,481` bytes,
  SHA-256
  `abc02804e17982a3be33675e4d471e91ea374e65b70167abc09e31acb412802d`;
- extracted `realesrgan-ncnn-vulkan.exe`: `6,161,408` bytes, SHA-256
  `07e49f7cbb4ede01ae4dd4c399d3a7e5846e3d2085c3128eff881e55cb7b1a0c`;
- the extracted anime x2 parameter and weight hashes matched the manifest.

`RealESRGAN_x2plus` did not run. Its exact authorized artifact is a PyTorch
checkpoint, while the current separate-process adapter accepts only a
checksum-bound NCNN `.param` and `.bin` pair. No conversion workflow has been
authorized, so the model remains adapter-incompatible and unapproved.

The existing `RealEsrganAdapter` ran `realesr-animevideov3` with scale 2 on two
fixed public synthetic seed-393 cases:

| Case | Input | Output | Adapter total | Inference | Result |
|---|---:|---:|---:|---:|---|
| `a1b41e74-1808-5dec-99c9-59f4c88f4004` | 1200 x 350 | 2400 x 700 | 978.6905 ms | 846.537 ms | PASS |
| `4b57ae01-eed3-55e3-bd47-cbae601ce431` | 1200 x 350 | 2400 x 700 | 916.0614 ms | 869.624 ms | PASS |

The repeated first case returned the adapter's verified cache hit in 72.9273
ms and reproduced output SHA-256
`988733f726e21e18cbc6fe17f28ba84ebcc32609a21c61aa237dc32e7b3ef6aa`.
The local environment was Windows `10.0.26200`, .NET SDK `10.0.302`, and an AMD
Radeon RX 6600 XT Vulkan device. The evidence JSON SHA-256 is
`26b8b7fa0ce02b3fbbbfc647bcb6993bacc6073a21841c07bbb4c2e46720e98e`.

This is a bounded adapter, runtime, dimension, and cache benchmark only. It
does not measure marker-center F1, shape/fill F1, numeric OCR exact match, axis
localization error, hallucinated structure rate, or peak memory. The NCNN
runtime also has no CPU or DirectML execution provider, and its transitive
redistribution notice inventory remains incomplete. The anime model therefore
remains experimental and is not approved for production or release bundling.

## Minimal NCNN runtime boundary

The official Windows archive also contains `vcomp140d.dll`, a Microsoft debug
OpenMP runtime that is not redistributable, plus unneeded demo media and model
families. None may enter a Graph Auto Reader package.

An ignored minimal runtime was assembled from exactly four required files:

| File | Bytes | SHA-256 |
|---|---:|---|
| `realesrgan-ncnn-vulkan.exe` | 6,161,408 | `07e49f7cbb4ede01ae4dd4c399d3a7e5846e3d2085c3128eff881e55cb7b1a0c` |
| `vcomp140.dll` | 182,704 | `8f72ef2e483465444b2059fc6744d6cb22cd8d8a27f6fa56befd2a42dcd0f78b` |
| `models/realesr-animevideov3-x2.param` | 3,173 | `b88ff4f00ebf019a7fdac17fdd45a7fd3665d37509efc5baf2e4da2e24420a04` |
| `models/realesr-animevideov3-x2.bin` | 1,247,368 | `548a36f9c3f4ab8da56cd3b13badf23968bee207b396dad14d04b830e5f2ab2d` |

The real adapter again succeeded on both fixed cases and verified its cache
using this minimal directory. Evidence SHA-256 is
`c480f97e09cf6117a86d5e8b565afa6e47aa3f85c14d9e46e3ef1932161fa967`.
This proves that `vcomp140d.dll`, demo media, and unused model families are not
runtime dependencies. It does not approve redistribution: the release
`vcomp140.dll` still needs an authorized Microsoft redistributable source and
maintainer license attestation.

The executable source closure also requires the complete
Real-ESRGAN-ncnn-vulkan MIT license, Tencent NCNN `LICENSE.txt`, libwebp
`COPYING`, and the embedded Toni Ronkko dirent MIT notice. The existing
Real-ESRGAN BSD notice covers the selected model payloads, not this runtime
closure.

## Verification method

Artifacts were downloaded from the exact official URLs above into ignored,
non-repository storage. File sizes and SHA-256 values were measured with
PowerShell `Get-FileHash -Algorithm SHA256`. Archive contents were inspected
with `Expand-Archive`. The synthetic set was generated with
`python -m ml.synthetic.generate --preset smoke --seed 393`. The actual
separate-process adapter was exercised through a local ignored harness. Only
metadata and fail-closed tests are written to the repository.
