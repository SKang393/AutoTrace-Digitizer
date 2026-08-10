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
| `realesr-animevideov3` NCNN x2 | `v0.2.5.0-ncnn-x2` | Official Real-ESRGAN Windows NCNN package plus unmodified Visual Studio 2022 VC Redist OpenMP runtime | BSD-3-Clause model notice; MIT/BSD/zlib runtime closure; Microsoft redistributable terms | Not bundled; retained only in ignored audit storage | Exact runtime, model, and Microsoft reference notices listed below | Package: `abc02804e17982a3be33675e4d471e91ea374e65b70167abc09e31acb412802d`; authorized `vcomp140.dll`: `55aba23cdcd6484fbb06f4155b8ca75adfce7a881f10afd0c49457165e677164` | Developer-only local adapter approved from checksum-bound execution and source-preservation evidence; scientific, memory, clean-machine, production, bundling, and release approvals remain false |

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
The pinned runtime source tree resolves NCNN to
`6125c9f47cd14b589de0521350668cf9d3d37e3c` and libwebp to
`8ea81561d2fdd382da60f57958741a7c23a18eb6`. Its embedded dirent header is Git
blob `f7a46dafcbf143ee8d0ac4b6a7d12b6fe28979e0`. Exact retained notices are:

- `LICENSES/Real-ESRGAN-NCNN-Vulkan-0.2.0-License.txt`, normalized SHA-256
  `5abb941454de437b0e90d78dcb72e3688f74e14bcd4e24393273cb5cd0e9c937`;
- `LICENSES/NCNN-6125c9f-License.txt`, normalized SHA-256
  `6495f972a09ad7f64ccd953e79adba91a93d862edc7135e6d95210bbf4002a01`;
- `LICENSES/libwebp-8ea81561-COPYING.txt`, normalized SHA-256
  `5aec868f669e384a22372a4e8a1a6cd7d44c64cd451f960ca69cc170d1e13acf`;
- `LICENSES/dirent-1998-2019-MIT-Notice.txt` for the embedded header attribution
  and MIT terms.

This closes the tracked source and notice inventory. It does not authorize
bundling because scientific, clean-machine, and production gates remain open.
The exact Microsoft binary has reviewed redistribution provenance only for the
checksum-bound authorized-vcomp profile.

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

The NCNN application source is MIT-licensed, NCNN carries BSD, zlib, and
additional embedded notices, libwebp is BSD-3-Clause, and dirent is MIT. Model
BSD metadata must not be treated as the runtime license audit; the separate
tracked notice files are required for any future package candidate.

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

This bounded adapter, runtime, dimension, and cache benchmark approves only
visibly labeled developer local evaluation of the exact manifest and runtime.
It does not measure marker-center F1, shape/fill F1, numeric OCR exact match, axis
localization error, hallucinated structure rate, or peak memory. The NCNN
runtime also has no CPU or DirectML execution provider. Its transitive runtime
provenance is now reviewed only for the exact authorized-vcomp profile. The anime model therefore
remains experimental and is not approved for production or release bundling.

## Minimal NCNN runtime boundary

The official Windows archive also contains `vcomp140d.dll`, a Microsoft debug
OpenMP runtime that is not redistributable, plus unneeded demo media and model
families. None may enter a Graph Auto Reader package.

An ignored minimal runtime was assembled from exactly four required files. The
Microsoft OpenMP file is an unmodified copy from the installed Visual Studio
2022 VC Redist profile named below, not the copy from the upstream archive:

| File | Bytes | SHA-256 |
|---|---:|---|
| `realesrgan-ncnn-vulkan.exe` | 6,161,408 | `07e49f7cbb4ede01ae4dd4c399d3a7e5846e3d2085c3128eff881e55cb7b1a0c` |
| `vcomp140.dll` | 193,152 | `55aba23cdcd6484fbb06f4155b8ca75adfce7a881f10afd0c49457165e677164` |
| `models/realesr-animevideov3-x2.param` | 3,173 | `b88ff4f00ebf019a7fdac17fdd45a7fd3665d37509efc5baf2e4da2e24420a04` |
| `models/realesr-animevideov3-x2.bin` | 1,247,368 | `548a36f9c3f4ab8da56cd3b13badf23968bee207b396dad14d04b830e5f2ab2d` |

The exact profile passed a direct public synthetic x2 process smoke with exit
code 0 in 2193.913 ms. Input SHA-256
`c548bb44965619566fcbfda86f44d092e2c8c71ecebe16b343e79e3b450d6eb4`
at 1200 x 350 produced SHA-256
`008076b47de90e10ff4b6cbd1efed043125f32b37089ca685055006f5865c4dc`
at exactly 2400 x 700. This proves only execution, exact 2x dimensions, and the
reduced inventory. It does not establish scientific fidelity.

The reviewed `vcomp140.dll` is version `14.44.35211.0`, has a valid Microsoft
Authenticode signature, and is byte-identical to
`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Redist\MSVC\14.44.35112\x64\Microsoft.VC143.OpenMP\vcomp140.dll`.
Its private authority attestation is ignored and Git-ineligible. The tracked
policy binds that attestation statement, the source `Redist.txt` hash, file
version, Authenticode signer and thumbprint, every runtime asset, the reduced
inventory, the direct smoke, and every notice checksum. Runtime redistribution
provenance is reviewed only for this exact profile.

The executable source closure is recorded in the manifest and exact tracked
notice files. The Real-ESRGAN BSD notice covers the selected model payloads;
the four upstream runtime notices cover the executable, NCNN, libwebp, and
dirent. `LICENSES/Microsoft-Visual-Cpp-2022-Redistribution-Reference.md`
records the Microsoft source and official terms references.

## Goal 21 manifest-driven local backend

The Goal 21 adapter probe now reads the exact tracked manifest before creating
an `IEnhancementService`. It verifies the model notice, executable checksum,
every parameter and weight checksum, Vulkan provider, runtime model name,
output scale, model redistribution flag, runtime redistribution status, and
benchmark approval. Distribution-purpose resolution fails closed. An explicit
local-evaluation purpose may return a service with a structured warning only
when the manifest's developer-only local-adapter approval is true. That
approval proves exact runtime compatibility, source preservation, and 2x output
dimensions, not scientific quality. A failed local approval returns
`MODEL_RUNTIME_INCOMPATIBLE` without a service. Distribution purpose remains
blocked while runtime redistribution and production benchmark approvals are
false.

The primary default remains `realesr-animevideov3` at output scale 2. A
manifest-driven run on public synthetic case
`a1b41e74-1808-5dec-99c9-59f4c88f4004` succeeded in 951.9313 ms total and
880.3792 ms inference time. The output was exactly 2400 x 700 with SHA-256
`988733f726e21e18cbc6fe17f28ba84ebcc32609a21c61aa237dc32e7b3ef6aa`.
The source hash was unchanged.

The same official archive contains the secondary
`RealESRGAN_x4plus_anime_6B` NCNN payload under runtime model name
`realesrgan-x4plus-anime`. This identity mapping is not inferred from the file
name. The pinned official Real-ESRGAN anime-model guide names
`RealESRGAN_x4plus_anime_6B` for PyTorch inference and, in the same usage
section, directs NCNN users to `-n realesrgan-x4plus-anime`:
<https://github.com/xinntao/Real-ESRGAN/blob/685d429c81888252bdb10f56c7754baededc3823/docs/anime_model.md#L20-L35>.
Exact local artifact evidence is:

| File | Bytes | SHA-256 |
|---|---:|---|
| `models/realesrgan-x4plus-anime.param` | 30,290 | `2b8fb6e0ae4d2d85704ca08c119a2f5ea40add4f2ecd512eb7f4cd44b6127ed4` |
| `models/realesrgan-x4plus-anime.bin` | 8,943,500 | `fe01c269cfd10cdef8e018ab66ebe750cf79c7af4d1f9c16c737e1295229bacc` |

Direct output-scale-2 runtime calls succeeded on both fixed seed-393 cases in
3359.2007 ms and 2286.0098 ms wall time, each producing exact 2400 x 700
output. The manifest-driven adapter then succeeded on the first case in
2220.2884 ms total and 2162.4602 ms inference time, with output SHA-256
`6e6c8b9c022b96cb98f82bddc23e1303a51732ba4d5862dcec044a3cc8efba5d`.
The source hash was unchanged.

A subsequent private Chandler run invalidated synthetic dimension success as
compatibility evidence. The 863 x 395 source produced an exact 1726 x 790 PNG
in 2124.1439 ms, but visual inspection showed that the output was cropped and
zoomed, losing substantial graph content. The source remained unchanged. The
failed output SHA-256 is
`d05e259e69f139d2649aaab8e99f866ccd4092534021612e93650b5048c97e85`.
The source and output remain private, ignored, and ineligible for Git.

Exact dimensions therefore do not establish scientific fidelity. The
secondary candidate is rejected for local adapter use and the manifest-driven
factory fails closed before process invocation. Small-text clarity,
open-circle preservation, filled-circle preservation, axis and tick clarity,
detector metrics, hallucinated-structure rate, and peak memory remain
unmeasured. No quality improvement is claimed. The primary manifest remains
the selected default identity and is approved only for visibly labeled
developer local evaluation. Its scientific and production approvals remain
false until the fixed quality gates pass.

Neither local result approves release bundling. Runtime redistribution
provenance is reviewed only for the checksum-bound authorized-vcomp profile.
Scientific fidelity, clean-machine execution, peak memory, and
production-quality benchmark approval remain mandatory. Enhancement is
optional, so the required offline CPU workflow continues with enhancement
disabled rather than claiming NCNN Vulkan has a CPU provider.

## Verification method

Artifacts were downloaded from the exact official URLs above into ignored,
non-repository storage. File sizes and SHA-256 values were measured with
PowerShell `Get-FileHash -Algorithm SHA256`. Archive contents were inspected
with `Expand-Archive`. The synthetic set was generated with
`python -m ml.synthetic.generate --preset smoke --seed 393`. The actual
separate-process adapter was exercised through a local ignored harness. Only
metadata and fail-closed tests are written to the repository.
