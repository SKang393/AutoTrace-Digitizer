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
| version | Pinned source revision `2870fa9244b0f0f69fb743fab1e08deefcb07b2b` |
| source | https://pdfium.googlesource.com/pdfium |
| license | BSD-3-Clause plus exact build-specific third-party notices |
| bundled or downloaded | Source-build profile is tracked; source, dependencies, binary, and review evidence remain ignored under `artifacts/pdfium-source/` |
| notice path | `LICENSES/PDFium-Minimal-Renderer-ThirdPartyNotices.txt`; 16 exact notice sources are checksum-bound to the 15-component target closure |
| checksum | Exact unbundled candidate SHA-256 `efd13a38cf3cd8e04d8284a42fff42923267293170424153b1a2a96dbf6fe8ea` |
| candidate build evidence | Two isolated Windows x64 static-ICU builds produced the byte-identical runner above and identical 240-label target-dependency and four-system-import evidence |
| dependency review | PASS for exact-binary license mapping: every target label maps once, NASM is build-only and unshipped, and the deterministic notice excludes unrelated source-tree licenses |
| compatibility patch | Checksum-bound patch `831cdc7351e06115e252a3aa3da9ce61d22b579fe19b1b3598d8b93f526bf5b6` adds the missing explicit Windows type include only while building; the official source checkout must be pristine afterward |
| privacy status | The reviewed runner reads PDF bytes from standard input and has no network dependency; the adapter writes only a temporary raw rendered page and deletes it after encoding |
| Git eligibility | The lock, minimal runner source, scripts, adapter, and tests are eligible; source checkout, native binary, approvals, and legal work product are not |
| review status | Dependency/license closure, ignored reviewed approval, public audit identity, and exact local backend execution pass for the exact candidate. Release bundling remains BLOCKED until the separate clean-machine and complete production workflow gates pass. |
