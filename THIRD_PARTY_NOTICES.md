# Third-Party Notices

<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

This audit covers the dependency and model set present on 2026-08-03. It does
not authorize a public release. A release remains blocked until every shipped
binary is present in the generated SBOM and checksum file and every row marked
blocked below is removed by new, independently verified evidence.

Current release blockers are:

1. `OpenCvSharpExtern.dll` has no complete source-audited inventory and notice
   set for its statically linked native dependencies.
2. No marker-center model has passed the production-runtime held-out gate.
3. No default super-resolution model and runtime pair has passed the required
   benchmark and packaging gates.
4. No OCR detector, OCR recognizer, or marker shape/fill classifier has an
   approved, checksum-pinned release artifact and schema-valid manifest.
5. PDFium has no selected binary, checksum, or build-specific notice set.

## Real-ESRGAN

- Component: official Real-ESRGAN model weights and/or inference code
- Copyright: Copyright (c) 2021, Xintao Wang
- License: BSD 3-Clause
- Full notice: `LICENSES/Real-ESRGAN-BSD-3-Clause.txt`
- Required action: preserve the complete notice in source and binary distributions.
- Do not imply endorsement by the copyright holder or contributors.

## Real-ESRGAN NCNN Vulkan

- Component: command-line inference runtime
- License: verify exact release and bundled dependency notices before shipment.
- Expected upstream licensing includes MIT-licensed application code and
  BSD-licensed NCNN components.
- Required action: record exact source tag, checksum, and all licenses.

## Microsoft .NET Desktop / WPF

- Component: self-contained Microsoft.NETCore.App and
  Microsoft.WindowsDesktop.App 10.0.10, published with SDK 10.0.302.
- Distribution terms: `LicenseRef-Microsoft-DotNet-Library` plus the exact
  Microsoft third-party notices.
- Notices: `LICENSES/DotNet-10.0.10-License.txt` and
  `LICENSES/DotNet-10.0.10-ThirdPartyNotices.txt`.
- Source files inspected on the build host: `C:\Program Files\dotnet\LICENSE.txt`
  SHA-256 `7f6839a61ce892b79c6549e2dc5a81fdbd240a0b260f8881216b45b7fda8b45d`
  and `ThirdPartyNotices.txt` SHA-256
  `deb4427a295e1ed474b0d81c5a0d972c1b550b9a715cda939cdfa9236b1b418f`.
- The repository copies are text-identical after newline normalization. Their
  SHA-256 values are `bb5bcf35bcb5ce9949455b07b8b60417cb5482872c01eaadbdbd15cf79e1cd47`
  and `2dc8f8c5a39401e928b5784ab564eb8b3ceb99ead3df8f260e0cab7e0bbecc7a`.
- Release condition: the packaged SBOM and checksums must confirm the same
  runtime version and exact shipped binaries.

## Imazen.WebP and libwebp

- Components: `Imazen.WebP` managed bindings and the Windows x64 libwebp runtime.
- Versions: Imazen.WebP 11.0.0; native runtime and libwebp 1.6.1.
- Sources: NuGet packages from Imazen's `libwebp-net` repository, commits
  `a2d53ed552b46e7f3a9a1a1f8ccd23e19f6f1595` and
  `462cd4a3bb76c171ff818cd16b0779614c3f8044`.
- Licenses: MIT for the managed bindings and BSD 3-Clause for libwebp.
- Notices: `LICENSES/Imazen.WebP-11.0.0-License.txt` and
  `LICENSES/libwebp-BSD-3-Clause.txt`.
- Native win-x64 payloads and SHA-256: `libsharpyuv.dll`
  `ceba59e40848e8e14960fc8c0f6ed125a6bfae216b3814c5ad202cb36d6a2a0d`,
  `libwebp.dll`
  `3de66254c8c0f47f299c7d55750589bf0e9fb6ab65778ed004a81e958ad1c024`,
  `libwebpdemux.dll`
  `af9124fc82821b7bef5d526d4dc4d857de7601833305f5d01ad68089706852c9`,
  and `libwebpmux.dll`
  `60947b85e84cb1e96f12e5863dab742641e51846713857063274eb93326672a6`.
- Use: bundled Windows x64 WebP decoding for immutable image import.


## OpenCV and OpenCvSharp

- Application dependency, but not approved for installer or portable
  distribution.
- Managed package: `OpenCvSharp4` 4.13.0.20260627, SHA-256
  `8acee778364e5eee6495d923732cacd8d895c7f683d2144f622b54418623d12c`.
- Native package: `OpenCvSharp4.runtime.win.slim` 4.13.0.20260627, SHA-256
  `281551a6c032d1aab316db9c1817bcded5a85188b24b2efd12c02665e7233817`.
- Upstream source revision: OpenCvSharp
  `b161e7e012f5101f6d5dc68a835c59db6cc88b18`; OpenCV
  `fe38fc608f6acb8b68953438a62305d8318f4fcd`.
- Managed and OpenCV source notices:
  `LICENSES/OpenCvSharp-4.13.0.20260627-License.txt` and
  `LICENSES/OpenCV-4.13.0-License.txt`. Package license metadata and these
  source licenses do not constitute a complete license inventory for the
  statically linked native DLL.
- Extracted native payload: `OpenCvSharpExtern.dll`, 55,547,904 bytes, SHA-256
  `1fa122bdb8e94175e7719fb8aaf2ab211268a756f5d0c7a13c710ed79ae30cd`.
- The published runtime ships no third-party notice bundle, SBOM, link map, or
  object-to-dependency map. Its actual scope also conflicts with the package
  description: the Windows slim workflow disables only contrib, `videoio`,
  `highgui`, and `dnn`; artifact inspection finds additional OpenCV modules and
  embedded build metadata reports non-free algorithms enabled.
- Release status: BLOCKED. Do not bundle this runtime until a source-audited
  win-x64 native build, exact dependency lock, binary hash, and complete
  third-party notice set are produced and verified.

## ONNX Runtime

- Component: Microsoft.ML.OnnxRuntime.DirectML 1.24.4, including CPU and
  DirectML execution providers.
- Source: Microsoft ONNX Runtime commit
  `2d924974ef147392ced8409d36bd6d2e7fcc8a74`.
- License: MIT with bundled third-party notices.
- Notices: `LICENSES/ONNX-Runtime-1.24.4-License.txt` and
  `LICENSES/ONNX-Runtime-1.24.4-ThirdPartyNotices.txt`.
- Transitive managed packages: Microsoft.ML.OnnxRuntime.Managed 1.24.4 and
  System.Numerics.Tensors 9.0.0. The latter's exact texts are stored at
  `LICENSES/System.Numerics.Tensors-9.0.0-License.txt` and
  `LICENSES/System.Numerics.Tensors-9.0.0-ThirdPartyNotices.txt`.

## Microsoft DirectML redistributable

- Component: Microsoft.AI.DirectML 1.15.4, transitively bundled by ONNX
  Runtime DirectML for Windows GPU execution.
- License: Microsoft DirectML redistributable terms permit distribution in
  applications that run on Windows and Xbox; third-party code is MIT/BSD.
- Notices: `LICENSES/DirectML-1.15.4-License.txt`,
  `LICENSES/DirectML-1.15.4-Code-License.txt`, and
  `LICENSES/DirectML-1.15.4-ThirdPartyNotices.txt`.
- Platform/privacy: Windows-only; no application telemetry is enabled. The
  exact redistributable license remains part of the release notice set.

## PaddleOCR / PaddleDetection

- Development candidates only: `PP-OCRv5_mobile_det` and
  `en_PP-OCRv4_mobile_rec`.
- Repository license evidence: Apache-2.0 at PaddleOCR tag `v3.5.0`, commit
  `33cbdd9deb2e00f61e7966db70669b249c005a37`.
- Bundled status: no PaddleOCR model archive, ONNX model, weight, dictionary,
  or runtime asset is distributed by this project.
- Release status: blocked until artifact-specific redistribution rights, an
  immutable source revision, exact SHA-256 checksums, conversion provenance,
  tensor contracts, provider compatibility, and benchmark results are
  independently verified and recorded in a schema-valid manifest.

## PdfPig

- Component: PdfPig 0.1.14.
- Source: `UglyToad/PdfPig` revision
  `88172af1c4d4f440949f59c94966c3880e3f6032`.
- NuGet SHA-256:
  `fe50ec7757adbb487bb52a30f1621384bbb5781fd0bff2169d9f18bc4bb91220`.
- License: Apache-2.0 plus the exact bundled PDFBox, Adobe AFM, and Adobe CMap
  external-component terms.
- Notice: `LICENSES/PdfPig-0.1.14-License.txt`.
- Release status: reviewed for local distribution when the exact package and
  notice above are staged.

## PDFium

- No PDFium package or binary is selected or bundled.
- Release status: BLOCKED for bundling until an exact build, binary checksum,
  redistribution review, and complete build-specific third-party notice set
  exist.

## IBM Plex Sans

- License: SIL Open Font License 1.1.
- Required action: preserve the OFL and follow reserved-font-name rules.
- No font file is distributed by the repository.

## Noto Sans

- License: SIL Open Font License 1.1.
- Required action: preserve the OFL.
- No font file is distributed by the repository.

## Original Graph Auto Reader models

### Marker center

- Model: `graph-marker-center` 0.1.0, `marker-center.onnx`, 37,542 bytes,
  SHA-256 `061a496167382d1bd11bb580bed383d2d1725da2001f9c440b7f1acc59ac116a`.
- License: Apache-2.0; notice:
  `models/manifest/markers/MARKER_CENTER_MODEL_NOTICE.md`.
- The local ignored artifact matches the manifest checksum, but generated
  weights are not tracked or staged.
- Release status: BLOCKED. The historical held-out report failed the one center
  per fixture and zero-mask legend rejection gates, and no authorized rerun
  exists after the production raw-mask max-gating repair.

### GraphSR x2 candidate

- Model: `graphsr-x2-candidate` 0.1.0-local-candidate,
  `graphsr-x2.onnx`, 207,110 bytes, SHA-256
  `4b0237683cd61ecd639015380bad9323a5fe79b295ffebf0c93720f51ef0d667`.
- License: Apache-2.0; notice:
  `models/manifest/graphsr/GRAPHSR_X2_CANDIDATE_NOTICE.md`.
- Manifest redistribution is false. The local ignored artifact matches the
  manifest checksum, but generated weights are not tracked or staged.
- Release status: BLOCKED. The candidate failed marker-center, thin-line, and
  open-marker fidelity thresholds and is not selected as the default.

## Missing production model classes

- PaddleOCR detector and recognizer candidates remain metadata-only. No model
  archive, dictionary, ONNX conversion, checksum, artifact-level redistribution
  approval, provider verification, benchmark, or schema-valid manifest exists.
- No marker shape/fill classifier artifact or model manifest exists.
- Release status: BLOCKED. Core offline digitization may not ship with guessed,
  missing, or unmanifested model bytes.

## Release audit table

| Item | Version | Source | License | SHA-256 | Bundled | Notice included | Reviewed |
|---|---|---|---|---|---|---|---|
| RealESRGAN_x2plus reference weight | v0.2.1 | Official Real-ESRGAN release | BSD-3-Clause | `49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb` | No | `LICENSES/Real-ESRGAN-BSD-3-Clause.txt` | Metadata yes; release blocked |
| realesr-general-x4v3 reference weight | v0.2.5.0 | Official Real-ESRGAN release | BSD-3-Clause | `8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292` | No | `LICENSES/Real-ESRGAN-BSD-3-Clause.txt` | Metadata yes; release blocked |
| realesr-animevideov3 NCNN x2 model package | v0.2.5.0 | Official Real-ESRGAN Windows NCNN package | BSD-3-Clause | `abc02804e17982a3be33675e4d471e91ea374e65b70167abc09e31acb412802d` | No | `LICENSES/Real-ESRGAN-BSD-3-Clause.txt` | Metadata yes; release blocked |
| NCNN runtime | v0.2.0 | Official Real-ESRGAN NCNN Vulkan release | MIT plus transitive notices pending | `1bbbdb12d470af80b035c773682e144c6c2f6ece9210832a289af0a48ce3fa9a` | No | No | No, bundling blocked |
| GraphSR x2 candidate | 0.1.0-local-candidate | Original Graph Auto Reader source | Apache-2.0 | `4b0237683cd61ecd639015380bad9323a5fe79b295ffebf0c93720f51ef0d667` | No | `models/manifest/graphsr/GRAPHSR_X2_CANDIDATE_NOTICE.md` | No, redistribution false and fidelity failed |
| Marker-center model | 0.1.0 | Original Graph Auto Reader source | Apache-2.0 | `061a496167382d1bd11bb580bed383d2d1725da2001f9c440b7f1acc59ac116a` | No | `models/manifest/markers/MARKER_CENTER_MODEL_NOTICE.md` | No, production held-out acceptance missing |
| OCR detector / recognizer | Not selected | No approved artifact | Unproven at artifact level | Not available | No | No | No, release blocked |
| Marker shape/fill classifier | Not selected | No approved artifact | Not reviewed | Not available | No | No | No, release blocked |
| ONNX Runtime DirectML | 1.24.4 | NuGet / microsoft/onnxruntime | MIT + notices | `57e9f11b73437bef7a309496135d4c1f96b1a8e9ddba60013fa27bfc1d788681` | Yes | `LICENSES/ONNX-Runtime-1.24.4-License.txt` | Yes |
| ONNX Runtime managed | 1.24.4 | NuGet / microsoft/onnxruntime | MIT + notices | `95cc5d366e876bcc9c39e87af277278aa3df56108fb572c884bf21f6b9e22182` | Yes | `LICENSES/ONNX-Runtime-1.24.4-License.txt` | Yes |
| Microsoft DirectML redistributable | 1.15.4 | NuGet / Microsoft.AI.DirectML | Microsoft redistributable + MIT/BSD notices | `4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9` | Yes | `LICENSES/DirectML-1.15.4-License.txt` and code license | Yes |
| System.Numerics.Tensors | 9.0.0 | NuGet / dotnet/runtime | MIT + notices | `b750243c36002a62b28b1ac5d3fbc284ad340ba1494cc36aca110611a0b1f959` | Yes | Exact license and third-party notices | Yes |
| OpenCvSharp managed bindings | 4.13.0.20260627 | NuGet / shimat/opencvsharp `b161e7e012f5101f6d5dc68a835c59db6cc88b18` | Apache-2.0 | `8acee778364e5eee6495d923732cacd8d895c7f683d2144f622b54418623d12c` | Yes | `LICENSES/OpenCvSharp-4.13.0.20260627-License.txt` | Yes, managed package only |
| OpenCvSharp Windows x64 slim runtime | 4.13.0.20260627 | NuGet / shimat/opencvsharp, OpenCV `fe38fc608f6acb8b68953438a62305d8318f4fcd` | Apache-2.0 plus linked native notices pending | `281551a6c032d1aab316db9c1817bcded5a85188b24b2efd12c02665e7233817` | Yes | OpenCvSharp and OpenCV source licenses only; native inventory pending | No, release blocked |
| .NET / WPF runtime | 10.0.10 | Microsoft .NET installed by SDK 10.0.302 | `LicenseRef-Microsoft-DotNet-Library` plus third-party notices | Per-file hashes required in release SBOM | Yes | Exact license and third-party notices | Source notice reviewed; artifact gate pending |
| Imazen.WebP | 11.0.0 | NuGet / imazen/libwebp-net | MIT | `f78a8f874f127bfa4688595950aa6292a8e20ea55fc2b60321523e1d005d5dff` | Yes | `LICENSES/Imazen.WebP-11.0.0-License.txt` | Yes |
| Imazen WebP native runtime win-x64 | 1.6.1 | NuGet / imazen/libwebp-net | MIT + libwebp BSD-3-Clause | `32df07f31f18b5f4e35409a73621d776d97761f4b601cbbbdc4efbacb6ab62f6` | Yes | Imazen MIT and libwebp BSD-3-Clause texts | Yes |
| PdfPig | 0.1.14 | NuGet / UglyToad/PdfPig `88172af1c4d4f440949f59c94966c3880e3f6032` | Apache-2.0 plus bundled external terms | `fe50ec7757adbb487bb52a30f1621384bbb5781fd0bff2169d9f18bc4bb91220` | Yes | `LICENSES/PdfPig-0.1.14-License.txt` | Yes |
| PDFium renderer | Not selected | No approved build | BSD-style plus build-specific notices | Not available | No | No | No, bundling blocked |

A release is blocked while any shipped row remains unreviewed or any required
core model class remains absent or unapproved.

## Goal 00 development and test packages

The following packages are used only to build or test the repository. They are
not application dependencies and are not included in either Windows
distribution skeleton. The full MIT text is in `LICENSES/MIT.txt`.

| Package | Version | Source | License | SHA-256 of NuGet package | Bundled | Notice | Reviewed |
|---|---:|---|---|---|---|---|---|
| Microsoft.NET.Test.Sdk | 18.8.1 | NuGet / microsoft/vstest | MIT | `8c1c5fcf73432dba471899e41ed0c342d1449e613dd981eb9e301916a18db895` | No | `LICENSES/MIT.txt` | Yes |
| MSTest.TestFramework | 4.3.3 | NuGet / microsoft/testfx | MIT | `77f28f595ebc26e2e554ab70757b5f2c6fed049fdde1663d0d56357ce061a734` | No | `LICENSES/MIT.txt` | Yes |
| MSTest.TestAdapter | 4.3.3 | NuGet / microsoft/testfx | MIT | `2701b1e104d3daffea29cb868353c2a80d383a7ac1e1e0d549449a3ba1be4e00` | No | `LICENSES/MIT.txt` | Yes |
| JsonSchema.Net | 8.0.5 | NuGet / json-everything | MIT | `cc54849aa7248aeb357dafefc61244328342fd1536e64177012b72330c4708f3` | No | `LICENSES/MIT.txt` | Yes |
| JsonPointer.Net | 6.0.1 | NuGet / json-everything | MIT | `888bce1f3ea38d3b069d7cb138aef5cf617c821db54e6412e00f522c1e2ef770` | No | `LICENSES/MIT.txt` | Yes |
| Json.More.Net | 2.2.0 | NuGet / json-everything | MIT | `1dbe3295c4ffe9b2c75400e99e3a0124ac93477714f968d895b94eefd03e4925` | No | `LICENSES/MIT.txt` | Yes |
| Humanizer.Core | 3.0.1 | NuGet / Humanizr | MIT | `5b1a9fd45457b6c42e94b16d6dfb1f62ee28666ac2b8e6408a431c6368ba0f0c` | No | `LICENSES/MIT.txt` | Yes |
| Microsoft.ApplicationInsights | 2.23.0 | NuGet / Microsoft | MIT | `e6c7f76e0ec26598c7b1e2be1777e839c84486655e8a878fc4c655bd2e918dbd` | No | `LICENSES/MIT.txt` | Yes |
| Microsoft.CodeCoverage | 18.8.1 | NuGet / microsoft/vstest | MIT | `beb0209f5f895c2e6c41cadaac5a6f3e6c641997feddbe7d24f9e9ccd95a2c57` | No | `LICENSES/MIT.txt` | Yes |
| Microsoft.Testing.Extensions.Telemetry | 2.3.3 | NuGet / microsoft/testfx | MIT | `0e8a35200c270e56eaca0d90ea9f3646e618010ed59099d8e0f9fd30366a08eb` | No | `LICENSES/MIT.txt` | Yes |
| Microsoft.Testing.Extensions.TrxReport.Abstractions | 2.3.3 | NuGet / microsoft/testfx | MIT | `6008074b70742f046f2077abdbce50b92ad808ee45712a405daccbe693470786` | No | `LICENSES/MIT.txt` | Yes |
| Microsoft.Testing.Extensions.VSTestBridge | 2.3.3 | NuGet / microsoft/testfx | MIT | `fd4970504230432359c5d0f48cd715df8649db468b7b6916413bc7dcb3881deb` | No | `LICENSES/MIT.txt` | Yes |
| Microsoft.Testing.Platform | 2.3.3 | NuGet / microsoft/testfx | MIT | `b07d00cab5f49f2d0981592044baef3d720a079e6941601c7790044584b3174a` | No | `LICENSES/MIT.txt` | Yes |
| Microsoft.Testing.Platform.MSBuild | 2.3.3 | NuGet / microsoft/testfx | MIT | `25e537fbc39e6ce0c0da4e8a4a8ae6d8fee98ae41945d4b586124a03075f4890` | No | `LICENSES/MIT.txt` | Yes |
| Microsoft.TestPlatform.ObjectModel | 18.8.1 | NuGet / microsoft/vstest | MIT | `e183005ae4f0c57d7111bedb6cec8b3e0bb12780326e703af9d41d2a5f0e6b21` | No | `LICENSES/MIT.txt` | Yes |
| Microsoft.TestPlatform.TestHost | 18.8.1 | NuGet / microsoft/vstest | MIT | `7875d837cd8da249c7817b72e97604532f5251f9af9b571059e3aeb3fa1b0093` | No | `LICENSES/MIT.txt` | Yes |
| MSTest.Analyzers | 4.3.3 | NuGet / microsoft/testfx | MIT | `780d8c8ac6ab5afc2e01d578f5676e61c9549c2fa159f1da4eddf9f54f827428` | No | `LICENSES/MIT.txt` | Yes |

JsonSchema.Net 9.x is intentionally excluded because its package license is no
longer MIT. Goal 00 pins the last reviewed MIT major line used by the schema
validation tests.
