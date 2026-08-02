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


## OpenCV and OpenCvSharp

- License: Apache-2.0 for current selected releases.
- Required action: verify native binary provenance and include exact notices.

## ONNX Runtime

- License: MIT.
- Required action: preserve notice and notices for any packaged execution provider.

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
| ONNX Runtime | TBD | TBD | MIT | TBD | TBD | No | No |
| OpenCV | TBD | TBD | Apache-2.0 | TBD | TBD | No | No |
| .NET 10 / WPF | TBD | official .NET distribution | MIT + notices | TBD | Yes | No | No |

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
