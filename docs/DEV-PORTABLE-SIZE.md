# Development Portable Size Audit

`packaging/Report-DevPortableSize.ps1` measures ignored development output
without changing it. The default JSON report is written to
`artifacts/dev-portable/size-report.json`, which remains ignored by Git.

## Report-only usage

Run the audit from the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File packaging/Report-DevPortableSize.ps1
```

The schema-version-1 report records:

- total repository bytes;
- development-portable build count and bytes;
- shared portable `Data` bytes;
- aggregate `bin` and `obj` bytes without double-counting nested output;
- `artifacts/goal19-opencv-source` bytes;
- the 20 largest directories;
- the exact build selected by `latest.json`;
- retained builds, cleanup candidates, pruned builds, and reclaimed bytes;
- scan warnings, including skipped reparse-point directories.

Report-only mode is the default and does not delete any build.

Development portable folders are ignored local builds. Every produced build
consumes the next central version even when application source is otherwise
unchanged. Timestamp and commit identity supplement the version; they do not
replace it. Dirty local-only builds are not permitted. See
`docs/VERSIONING-AND-RELEASES.md`.

## Explicit pruning

The current retention policy keeps exactly the build selected by `latest.json`.
After the complete ledger is committed and confirmed on `origin/main`, apply
the same validated retention path used by the watcher:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File packaging/Watch-DevPortable.ps1 -ApplyRetentionOnce
```

The watcher preserves the exact `latest.json` target and deletes only validated,
non-reparse-point immediate child directories under
`artifacts/dev-portable/builds`. Missing, malformed, outside-root, incomplete,
unpushed, or nested metadata stops pruning before any build is removed. A failed
push retains both the previous and new build directories.

Do not prune while a build or validation run is writing the portable output.
Keep any build or evidence referenced by a current readiness report.

## Goal 21 baseline

The report-only baseline generated at `2026-08-04T02:02:49.0258564+00:00`
measured the following state. No cleanup was requested or performed.

| Metric | Bytes | Size |
| --- | ---: | ---: |
| Goal 19 worktree | 6,244,452,084 | 5.816 GiB |
| 14 development portable builds | 2,858,282,042 | 2.662 GiB |
| Shared portable `Data` | 0 | 0 GiB |
| 63 top-level `bin`/`obj` directories | 928,669,602 | 0.865 GiB |
| OpenCV audit workspace | 2,111,771,929 | 1.967 GiB |

The exact preserved latest build was
`artifacts/dev-portable/builds/0.0.20-20260804T015043887Z-a2074374`.
All 14 builds were recorded as retained, with zero prune candidates, zero
pruned builds, and zero reclaimed bytes because the run stayed in report-only
mode.

The largest directories at that checkpoint were:

| Rank | Directory | MiB |
| ---: | --- | ---: |
| 1 | `artifacts` | 5,115.05 |
| 2 | `artifacts/dev-portable` | 2,920.61 |
| 3 | `artifacts/dev-portable/builds` | 2,725.87 |
| 4 | `artifacts/goal19-opencv-source` | 2,013.94 |
| 5 | `tests` | 564.11 |
| 6 | `artifacts/goal19-opencv-source/sources` | 397.46 |
| 7 | `artifacts/goal19-opencv-source/sources/opencv` | 288.13 |
| 8 | `src` | 268.90 |
| 9 | `src/GraphReader.App` | 255.00 |
| 10 | `src/GraphReader.App/bin` | 250.60 |
| 11 | `artifacts/goal19-opencv-source/evidence-repro-pass2-final-a` | 223.46 |
| 12 | `artifacts/goal19-opencv-source/evidence-repro-pass2-final-b` | 223.46 |
| 13 | `artifacts/goal19-opencv-source/evidence-repro-fixed-b` | 223.44 |
| 14 | `artifacts/goal19-opencv-source/evidence-repro-pass2-a` | 223.44 |
| 15 | `artifacts/goal19-opencv-source/evidence-repeat` | 223.38 |
| 16 | `artifacts/goal19-opencv-source/evidence-repro-fixed-a` | 223.35 |
| 17 | `artifacts/goal19-opencv-source/evidence` | 222.97 |
| 18 | `tests/GraphReader.Integration.Tests` | 212.28 |
| 19 | `tests/GraphReader.Integration.Tests/bin` | 209.53 |
| 20 | `artifacts/dev-portable/builds/0.0.20-20260804T015043887Z-a2074374` | 194.72 |

The ignored JSON report is the machine-readable evidence for the complete
schema and exact byte values.

## Verification

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File packaging/tests/Test-DevPortableSize.Tests.ps1
```

The focused test verifies the report schema and measured fixture sizes,
report-only no-delete behavior, rejection of the retired direct-pruning path,
and traversal safety. Push-gated retention is covered by
`packaging/tests/Test-DevPortable.Tests.ps1`.
