<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# OpenCV source-audit fallback

This directory is a fail-closed source-build scaffold for the Windows x64
`OpenCvSharpExtern.dll` used by Graph Auto Reader's deterministic axis stage.
It does not approve a binary for public release and does not modify the public
release audit.

The profile pins:

- OpenCvSharp commit `b161e7e012f5101f6d5dc68a835c59db6cc88b18`;
- OpenCV commit `fe38fc608f6acb8b68953438a62305d8318f4fcd`;
- vcpkg commit `1e199d32ad53aab1defda61ce41c380302e3f95c`;
- Visual Studio, MSVC, Windows SDK, CMake, architecture, CRT, and Release
  configuration in `source-lock.json`;
- only OpenCV `core`, `imgproc`, and `imgcodecs` modules;
- nonfree, contrib, video, UI, codec, GPU, network-downloaded, and unnecessary
  integrations disabled.

`imgcodecs` remains in the native wrapper because the pinned OpenCvSharp source
exports unguarded image-codec entry points. All external image codecs are OFF.
Graph Auto Reader itself currently calls only `core` and `imgproc` operations.

## Commands

From the repository root:

```powershell
powershell -File packaging/opencv-source/Initialize-SourceCheckouts.ps1
powershell -File packaging/opencv-source/Build-SourceAuditedOpenCvSharp.ps1 -Phase Preflight
powershell -File packaging/opencv-source/Build-SourceAuditedOpenCvSharp.ps1 -Phase Configure
powershell -File packaging/opencv-source/Build-SourceAuditedOpenCvSharp.ps1 -Phase Build -Jobs 8
powershell -File packaging/opencv-source/Build-SourceAuditedOpenCvSharp.ps1 -Phase Collect
powershell -File packaging/opencv-source/Test-SourceAuditEvidence.ps1
powershell -File packaging/opencv-source/tests/Test-OpenCvSourceAudit.ps1
powershell -File packaging/opencv-source/Compare-SourceBuilds.ps1 -FirstEvidenceRoot <first> -SecondEvidenceRoot <second>
```

Sources and build evidence are written only under ignored
`artifacts/goal19-opencv-source/` paths by default.

## Completion gate

The mechanical collection phase deliberately writes:

- `dependency-inventory.json` with `reviewStatus` set to `requires-review`;
- `third-party-notices.candidate.txt` with an incomplete warning;
- CMake caches and dependency graphs;
- the MSVC linker map;
- the PE import report;
- copies of the exact lock, initial cache, and triplet inputs;
- the built DLL and a SHA-256 manifest.

The evidence validator fails until every linked library and imported DLL has a
source, license, reviewed status, and notice disposition, and an independently
reviewed `third-party-notices.reviewed.txt` declares `REVIEW STATUS: COMPLETE`.
The candidate notice must never be copied into a release as if it were complete.
It also requires exact inventory coverage of every linker-map library and PE
import, and exactly one SHA-256 entry for every retained evidence file.

The build disables variable OpenCV build timestamps, uses a canonical embedded
install prefix, canonicalizes OpenCV's generated build-directory metadata, maps
each isolated evidence root to one canonical MSVC source path with
`/experimental:deterministic /pathmap`, installs into the selected evidence root
with CMake's `--prefix` override, and passes MSVC `/Brepro`. Cross-root
reproducibility is accepted only when
`Compare-SourceBuilds.ps1` finds byte-identical DLL and linker-map hashes with
identical retained inputs.

Replacing the current NuGet runtime additionally requires functional parity,
axis benchmarks, a second clean-build binary hash comparison, clean-machine
tests, and maintainer approval outside this scaffold.
