# Third-Party Notices

<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

This file is a planning template. Before every release, replace versions,
checksums, and packaging status with the exact shipped values.

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

- Component: self-contained .NET 10 Windows desktop runtime and WPF libraries
- License: MIT plus the exact third-party notices shipped with the selected
  .NET runtime.
- Required action: preserve the complete notice set generated for the exact
  published runtime.

## Imazen.WebP and libwebp

- Components: `Imazen.WebP` managed bindings and the Windows x64 libwebp runtime.
- Versions: Imazen.WebP 11.0.0; native runtime and libwebp 1.6.1.
- Sources: NuGet packages from Imazen's `libwebp-net` repository, commits
  `a2d53ed552b46e7f3a9a1a1f8ccd23e19f6f1595` and
  `462cd4a3bb76c171ff818cd16b0779614c3f8044`.
- Licenses: MIT for the managed bindings and BSD 3-Clause for libwebp.
- Notices: `LICENSES/MIT.txt` and `LICENSES/libwebp-BSD-3-Clause.txt`.
- Use: bundled Windows x64 WebP decoding for immutable image import.


## OpenCV and OpenCvSharp

- License: Apache-2.0 for current selected releases.
- Required action: verify native binary provenance and include exact notices.

## ONNX Runtime

- Component: Microsoft.ML.OnnxRuntime.DirectML 1.24.4, including CPU and
  DirectML execution providers.
- Source: Microsoft ONNX Runtime commit
  `2d924974ef147392ced8409d36bd6d2e7fcc8a74`.
- License: MIT with bundled third-party notices.
- Notices: `LICENSES/ONNX-Runtime-1.24.4-License.txt` and
  `LICENSES/ONNX-Runtime-1.24.4-ThirdPartyNotices.txt`.
- Transitive managed packages: Microsoft.ML.OnnxRuntime.Managed 1.24.4 and
  System.Numerics.Tensors 9.0.0. The latter's exact notice is stored at
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

- License: Apache-2.0.
- Required action: record the exact exported model source, version, checksum,
  training modifications, and applicable model notice.

## PdfPig

- License: Apache-2.0.
- Required action: preserve notice.

## PDFium

- License: BSD-style with additional third-party notices.
- Required action: ship the complete notice set for the exact binary build.

## IBM Plex Sans

- License: SIL Open Font License 1.1.
- Required action: preserve the OFL and follow reserved-font-name rules.
- This planning kit does not distribute font files.

## Noto Sans

- License: SIL Open Font License 1.1.
- Required action: preserve the OFL.
- This planning kit does not distribute font files.

## Release audit table

| Item | Version | Source | License | SHA-256 | Bundled | Notice included | Reviewed |
|---|---|---|---|---|---|---|---|
| Real-ESRGAN model | TBD | TBD | BSD-3-Clause | TBD | TBD | No | No |
| NCNN runtime | TBD | TBD | TBD | TBD | TBD | No | No |
| ONNX Runtime DirectML | 1.24.4 | NuGet / microsoft/onnxruntime | MIT + notices | `57e9f11b73437bef7a309496135d4c1f96b1a8e9ddba60013fa27bfc1d788681` | Yes | `LICENSES/ONNX-Runtime-1.24.4-License.txt` | Yes |
| ONNX Runtime managed | 1.24.4 | NuGet / microsoft/onnxruntime | MIT + notices | `95cc5d366e876bcc9c39e87af277278aa3df56108fb572c884bf21f6b9e22182` | Yes | `LICENSES/ONNX-Runtime-1.24.4-License.txt` | Yes |
| Microsoft DirectML redistributable | 1.15.4 | NuGet / Microsoft.AI.DirectML | Microsoft redistributable + MIT/BSD notices | `4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9` | Yes | `LICENSES/DirectML-1.15.4-License.txt` and code license | Yes |
| System.Numerics.Tensors | 9.0.0 | NuGet / dotnet/runtime | MIT + notices | `b750243c36002a62b28b1ac5d3fbc284ad340ba1494cc36aca110611a0b1f959` | Yes | `LICENSES/System.Numerics.Tensors-9.0.0-ThirdPartyNotices.txt` | Yes |
| OpenCV | TBD | TBD | Apache-2.0 | TBD | TBD | No | No |
| .NET 10 / WPF | TBD | official .NET distribution | MIT + notices | TBD | Yes | No | No |
| Imazen.WebP | 11.0.0 | NuGet / imazen/libwebp-net | MIT | `f78a8f874f127bfa4688595950aa6292a8e20ea55fc2b60321523e1d005d5dff` | Yes | `LICENSES/MIT.txt` | Yes |
| Imazen WebP native runtime win-x64 | 1.6.1 | NuGet / imazen/libwebp-net | MIT + libwebp BSD-3-Clause | `32df07f31f18b5f4e35409a73621d776d97761f4b601cbbbdc4efbacb6ab62f6` | Yes | `LICENSES/libwebp-BSD-3-Clause.txt` | Yes |

A release is blocked while any shipped row remains unreviewed.

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
