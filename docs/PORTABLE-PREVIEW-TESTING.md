<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Development portable preview testing

The development portable is a local maintainer preview. It is visibly marked
`Development Preview`, defaults to the real-data `ManualPreview` runtime, and
must not be uploaded or redistributed while production model and native
provenance gates remain unresolved.

It does not change or bypass `packaging/Build-Windows.ps1`,
`packaging/Test-ReleaseArtifact.ps1`, or the public release audit.

## Build once

From the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\packaging\Build-DevPortable.ps1 `
  -AllowDirty `
  -FastTestsOnly
```

Omit `-AllowDirty` for a stable checkpoint. Omit `-FastTestsOnly` to run the
full Release suite, public synthetic scoreboard, packaging regression, and
localization regression before publishing the new local preview.

## Watch and rebuild

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\packaging\Watch-DevPortable.ps1 `
  -BuildOnStart `
  -FastTestsOnly `
  -AllowDirty `
  -LaunchAfterBuild
```

The watcher debounces relevant changes for two seconds and permits one build at
a time. Changes raised during a build queue one additional build. Existing WPF
processes are never modified or overwritten.

## Launch the latest successful preview

```powershell
.\packaging\Run-Latest-DevPortable.cmd
```

The launcher reads `artifacts\dev-portable\latest.json`, verifies the required
SHA-256 of the selected executable, launches its immutable build, and sets
`GRAPHREADER_DEV_PORTABLE_DATA_ROOT` to the shared local root. Missing,
malformed, or mismatched executable checksums fail before process start:

```text
artifacts\dev-portable\Data
```

The application accepts that override only when `portable.mode` is beside the
executable. A normal portable release without the override continues to use
`.\Data`.

## Manual workflow check

1. Confirm the window says `Development Preview`, the version, short commit,
   `ManualPreview`, and lists unavailable automatic stages.
2. Import a private graph image through the application file picker. Do not add
   the image to Git.
3. Confirm the original image is shown in a tab and zoom, pan, and the fixed
   magnifier remain usable.
4. Set the three manual anchors `(1, 0)`, `(1, yMax)`, and `(xMax, 0)`.
5. Create a series with an explicit shape and fill.
6. Add points, move one, delete one, and reassign one when multiple series are
   present.
7. Add, move, label, and delete a phase divider as needed.
8. Save the `.garproj`, close it, reopen it, and confirm edits persist.
9. Export intervention-specific CSV and confirm the header is
   `x_value,y_value,phase`.
10. Confirm the source SHA-256 is unchanged.

Store private screenshots, logs, and feedback only under ignored paths such as
`artifacts\dev-portable\feedback`.

## Local portable path validation

Run the bounded harness after a stable preview build:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\packaging\portable-validation\Test-PortableCleanProfile.ps1
```

The harness is fail-closed and records JSON evidence under
`artifacts\portable-validation\`. It exercises a space and Korean Unicode path,
the shared preview Data root, normal `.\Data`, process-owned endpoint polling,
the application registry key, the read-only WPF diagnostic, and selected-root
file-system tracing. It is a local isolated-profile simulation, not clean-user
or clean-VM evidence. See `packaging/portable-validation/README.md` for exact
evidence boundaries.

## Failure behavior

A failed build writes `artifacts\dev-portable\last-failure.json` and leaves the
previous successful `latest.json` unchanged. Every successful build folder
under `artifacts\dev-portable\builds` is immutable and receives a new version,
UTC timestamp, and short-commit name.
