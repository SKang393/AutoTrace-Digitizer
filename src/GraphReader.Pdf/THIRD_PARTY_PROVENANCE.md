<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# GraphReader.Pdf dependency provenance

## PdfPig

| Field | Value |
|---|---|
| dependency/model | PdfPig |
| version | 0.1.14 |
| source | https://www.nuget.org/packages/PdfPig/0.1.14 |
| source revision | 88172af1c4d4f440949f59c94966c3880e3f6032 |
| license | Apache-2.0 |
| bundled or downloaded | Restored through NuGet and bundled with the application publish output |
| notice path | `THIRD_PARTY_NOTICES.md`, PdfPig section; package license expression `Apache-2.0` |
| package SHA-256 | FE50EC7757ADBB487BB52A30F1621384BBB5781FD0BFF2169D9F18BC4BB91220 |
| privacy status | Local PDF content is processed in memory; the dependency has no network requirement |
| Git eligibility | Package binary remains in the NuGet cache and build output; only this provenance record is committed |
| review status | Approved for Session 04 development and Apache-2.0 distribution; exact notice staging remains Session 17 work |

## PDFium backend

| Field | Value |
|---|---|
| dependency/model | PDFium renderer backend |
| version | Not selected |
| source | Not selected |
| license | BSD-style plus build-specific third-party notices required |
| bundled or downloaded | Not bundled in Session 04 |
| notice path | `THIRD_PARTY_NOTICES.md`, PDFium section |
| checksum | Not available because no binary is selected |
| privacy status | The adapter contract requires local in-process rendering and no network access |
| Git eligibility | No backend binary is eligible until exact build provenance and the complete notice set are reviewed |
| review status | BLOCKED for bundling; Session 04 uses the injected adapter boundary and deterministic test backend |
