<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Goal 21 completion

## Problem

The 0.0.20 manual preview needed a practical graph-first layout, predictable
fit and zoom behavior, visible manual editing tools, an evidence-backed local
enhancement path, and bounded development-output growth. None of that could
weaken the production release audit or imply automatic detector accuracy.

## Real-ESRGAN integration

The official `realesr-animevideov3` NCNN model is the default only when an
explicit ignored local runtime and tracked manifest are configured. The
manifest-driven factory verifies runtime, dependency, parameter, weight, and
notice checksums before exposing the service. The application stores a derived
2x PNG under the active application cache, leaves the original immutable, and
records the envelope and selected Original, Enhanced, or Comparison mode in
project provenance.

The local runtime and weights are not copied into the development portable.
The launcher supplies their paths only to the child process and clears inherited
paths when enhancement is disabled. Missing, malformed, incompatible, or
checksum-invalid inputs return localized fail-closed diagnostics.

The primary Chandler output retained the complete graph frame, open-circle
holes, filled markers, axes, ticks, and small labels. The secondary
`RealESRGAN_x4plus_anime_6B` NCNN output had exact 1726 by 790 dimensions but
cropped and magnified the scientific content. Its manifest records
`failed_scientific_fidelity`, `local_adapter_approval: false`, and the factory
rejects it before runtime discovery.

## UI refinement

The native WPF shell now provides:

- a top graph tab strip with dirty markers, close controls, and keyboard tab
  navigation;
- a top project summary;
- the series list and series editor in the left panel;
- a wider, resizable right inspector with fail-soft width persistence;
- always-visible calibration, series, filled/open point, point move/delete,
  divider add/move/delete/label, fit, zoom, and reset controls;
- original, enhanced, and side-by-side comparison views while editing remains
  bound to the immutable original pixel space;
- initial-load, resize, and maximize fit recalculation plus button and wheel
  zoom below fit scale;
- original-pixel coordinate emission at non-96-DPI source metadata;
- dirty-tab close protection and relation sanitation after series-role edits.

## Size audit

`packaging/Report-DevPortableSize.ps1` reports repository, portable build,
shared Data, bin/obj, OpenCV audit, and largest-directory sizes. Report-only is
the default. Explicit pruning validates the cleanup root, preserves the exact
`latest.json` target, can retain fallback builds, rejects traversal and reparse
points, and supports `-WhatIf`.

The recorded Goal 21 baseline was 6,244,452,084 bytes, with 14 development
portable builds totaling 2,858,282,042 bytes, zero shared Data bytes,
928,669,602 bin/obj bytes, and 2,111,771,929 OpenCV audit bytes. Nothing was
pruned. The final report after the clean 0.0.21 checkpoint measured
6,530,103,082 repository bytes, with 15 builds totaling 3,062,654,186 bytes,
zero shared Data bytes, 1,006,173,554 bin/obj bytes, and the exact 0.0.21 build
retained as latest. Nothing was pruned.

## Files changed

- `Directory.Build.props`
- `src/GraphReader.App/` WPF shell, canvas, composition, localization,
  workspace service, and view models
- `src/GraphReader.SuperResolution/` manifest-driven backend, adapter
  contracts, and benchmark selection
- `models/manifest/super-resolution/` primary and rejected secondary manifests
  plus provenance audit
- `packaging/Run-Latest-DevPortable.ps1`
- `packaging/Report-DevPortableSize.ps1`
- App, integration, super-resolution, and packaging tests
- `docs/DEV-PORTABLE-SIZE.md`, `docs/1.0-READINESS.md`, and this record

Private Chandler input, generated derivatives, screenshots, runtime binaries,
model weights, build output, and size reports remain ignored and untracked.

## Tests and commands

- Full Release solution: 585 passed, 4 authorized opt-in local-asset tests
  skipped, 0 failed.
- WPF App suite: 40/40 passed.
- Super-resolution suite: 70 passed, 1 authorized local-runtime test skipped.
- Opt-in Chandler WPF and Real-ESRGAN validation: 1/1 passed.
- Development-portable packaging regression: 7/7 passed.
- Release-artifact packaging regression: 43/43 passed.
- Size-report tests: 4/4 in Windows PowerShell and 4/4 in PowerShell 7.
- Localization self-tests: 9/9 passed; repository audit has 177 keys with zero
  missing, extra, duplicate, or unresolved references.
- Public synthetic scoreboard: 37/37 gates passed in 211.419 ms. This is metric
  contract smoke, not detector accuracy.
- Clean `Build-Windows.ps1 -AuditOnly` checkpoint: `ReleaseReady=False`, 7
  manifests, 0 redistributable model files, 13 blockers, and no artifacts
  emitted.
- The exact 0.0.21 executable launched from the clean implementation build as
  process 33160 with its shared portable Data root and ignored local-only
  enhancement configuration.
- `git diff --check`: passed.

## Metrics and timing

- Chandler source: 863 by 395 pixels; SHA-256
  `65bd6e1021bd9a3f2d2564a640ff90060a34e784782f2d847b4b26f9c6c342e1`
  before and after all work.
- Primary x2 output: 1726 by 790 pixels; SHA-256
  `954090b7f3b0783123f51b5a523672fdc889ab5e13b1dc935b5818f05b4729d1`;
  945.0486 ms in the final private pre-commit run.
- Secondary rejected output: 1726 by 790 pixels; SHA-256
  `d05e259e69f139d2649aaab8e99f866ccd4092534021612e93650b5048c97e85`;
  2124.1439 ms; source unchanged but graph content cropped.
- Final private WPF evidence before checkpoint build: 1400 by 900 pixels;
  SHA-256 `6cf61c261b0d8989a197246917dc5a12f3530e7dbd95460b62b2ab8d7caedff6`.
- Final development output report: 6.082 GiB repository, 2.852 GiB portable
  builds, 0.937 GiB bin/obj, and 1.967 GiB OpenCV audit workspace.

## License review

No runtime binary, model weight, private graph, or generated output was added to
Git or the portable payload. The primary model's exact official artifact and
BSD-3-Clause notice are recorded, but the NCNN runtime and Microsoft
`vcomp140.dll` redistribution evidence remain incomplete. Local execution does
not imply public redistribution approval. The rejected secondary is retained as
audit metadata only. No GPL, AGPL, SSPL, BUSL, or non-commercial dependency was
introduced.

## Portable artifact

- exact executable path: `C:\Users\Sungwoo\Desktop\Graph-auto-reader-goal19\artifacts\dev-portable\builds\0.0.21-20260804T025218204Z-fa299729\GraphReader.App.exe`
- executable SHA-256: `569db2313e4d5ace5022e95778bf767dd6dcce64b6d553b138d883f2fcdff4ce`
- build folder: `C:\Users\Sungwoo\Desktop\Graph-auto-reader-goal19\artifacts\dev-portable\builds\0.0.21-20260804T025218204Z-fa299729`
- version: `0.0.21`
- implementation commit: `fa299729581ea625a39a1a6f56c29e2196ef77a8`
- portable marker present; no NCNN runtime, Real-ESRGAN parameter, or model
  weight payload is present
- default enhancement model: `realesr-animevideov3` x2, explicit ignored local
  evaluation only, not bundled
- automatic stages still unavailable for production: enhancement, axis, OCR,
  markers, legends, and phases

## Known limitations

- Automatic production digitization remains unavailable and unvalidated on the
  supplied Chandler graph.
- The primary enhancement has bounded preservation evidence, not a
  release-eligible accuracy benchmark, CPU fallback, or clean-machine proof.
- The secondary comparison candidate is scientifically incompatible and cannot
  be selected.
- No release installer or portable ZIP exists, and no clean Windows VM parity,
  repair, upgrade, uninstall, or offline workflow has run.
- Windows Computer Use could list the live WPF window but could not bind it for
  post-build input because its helper reported the same OneDrive owner as both
  stale and current. The pre-checkpoint private WPF capture and automated WPF
  integration remain the direct layout evidence; no post-build input claim is
  made.
- OpenCvSharp native and NCNN runtime redistribution approvals remain blocked.
- The size audit intentionally retained all historical builds; cleanup requires
  explicit opt-in.

## Acceptance status

Manual UI workflow: PASS

Real-ESRGAN integration: PASS

Development-portable size audit: PASS

Automatic production workflow: FAIL

1.0 readiness: FAIL
