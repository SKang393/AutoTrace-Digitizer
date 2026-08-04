<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Goal 20 completion record

## Problem

Goal 19 produced a real manual-first WPF portable preview, production audit
groundwork, and model experiments, but it could not be committed because the
mandatory maintainer graph was unavailable and the portable write audit treated
attributed Windows, WPF, .NET, and GPU cache activity as Graph Auto Reader
persistence failures.

## Recovered Goal 19 work

Before changing the Goal 19 worktree, recovery artifacts were recorded under
the ignored `.codex/goal20-recovery/` directory:

- `goal19-working-tree.patch`: 205,245 bytes and nonempty;
- `goal19-working-files.zip`: 269,870 bytes and nonempty;
- `status-before.txt`, `diff-stat-before.txt`, and `working-files.list`;
- zero recovery or private-data files tracked by Git.

No reset, clean, checkout, or unrelated original-workspace change was made.

## Real graph manual validation

The user-supplied Chandler PNG was copied byte for byte to the ignored private
path `data/private/golden/chandler.png`. Its SHA-256 before import, after the
workflow, and in the retained source file is
`65bd6e1021bd9a3f2d2564a640ff90060a34e784782f2d847b4b26f9c6c342e1`.
It is private local evaluation data, is not training data, and is not eligible
for Git or release packaging.

The opt-in STA test constructs the real `ManualPreview` application
composition, actual `MainWindow`, actual `MainWindowViewModel`, and real manual
workspace service. It does not use recorded detections or fake graph data. The
passing workflow directly verifies:

- real PNG import and immutable source bytes;
- graph tab, zoom in/out, horizontal pan, and fixed right magnifier;
- three-anchor calibration at `(1,0)`, `(1,100)`, and `(24,0)`;
- filled-circle and open-circle series;
- four filled points and two open generalization probes;
- point move, delete, re-add, and updated counts;
- phase dividers between sessions 9/10 and 14/15 with visible labels;
- Save As, existing-path Save, close, and reopen persistence;
- autosave under portable `Data` and recovery to a new project;
- GraphReader.Export output with exact header `x_value,y_value,phase`;
- probe inclusion using `g`, with no fabricated automatic marker or OCR rows;
- a rendered 1400 by 900 WPF screenshot.

Parent rerun result: 1/1 passed. Evidence is retained in the
ignored directory
`artifacts/goal20-chandler/run-20260804T011803056Z-b077abbe6df549f783dc6728b6f53c56/`.
The screenshot is 103,264 bytes with SHA-256
`28002c8074aa3f2310c99c214f17adfb082ca4e8df724b280c297f4312329bf4`.
The report explicitly records `ManualPreview`, `Portable`, `usesFakeGraphData:
false`, and `automaticDetectionAccuracyClaimed: false`.

The actual packaged WPF executable was also launched and observed responsive.
The Windows computer-control helper could list the window but could not retain
its owner identity for interaction. This helper limitation is recorded rather
than treated as manual click evidence. The direct workflow evidence comes from
the real WPF composition and rendered window described above.

## Portable isolation classification

The portable harness now classifies mutations by process, responsible
component, destination, and purpose. Graph Auto Reader settings, cache, logs,
autosave, recovery, or project persistence outside configured `Data` or an
explicit user-selected writable path fail. Attributed Windows, WPF, .NET, font,
Direct3D, or GPU-driver caches remain warnings with process, PID, executable,
component, path, purpose, and evidence.

Both PowerShell hosts passed 10/10 self-tests. The negative tests create
initially absent LocalAppData and RoamingAppData `GraphAutoReader` roots and
prove that application-owned writes fail. Exact roots are SHA-256 and metadata
snapshotted before and after all three scenarios, so initially absent root
creation cannot escape observation. The live harness passed 7/7 gates with 55
allowed `Data` events, 12 attributed external warnings, zero
application-owned or unattributed failures, zero watcher errors, three network
samples, and zero observed TCP or UDP endpoints. Evidence is retained at
`artifacts/portable-validation/20260804T014803963Z-ace36c81/portable-clean-profile-report.json`.

This remains local isolated-profile evidence, not clean-profile or clean-VM
evidence. FileSystemWatcher does not directly report a writer PID, and endpoint
sampling can miss a very brief connection.

## Implementation

- Recovered the complete uncommitted Goal 19 implementation without weakening
  any release gate.
- Added an opt-in real-image WPF integration validation that fails closed when
  private input is not explicitly configured.
- Corrected manual Generalization and Maintenance phase semantics to `g` and
  `m`, including reassignment, persistence, audit export, and minimal export.
- Replaced root-only portable-write classification with purpose-aware,
  process/component-evidenced classification, exact external application-root
  snapshots, and required negative cases.
- Neutralized spreadsheet formula prefixes in CSV text fields while preserving
  numeric scientific values and unmodified structured JSON data.
- Promoted only the internal checkpoint version from `0.0.19` to `0.0.20`.

## Files changed

- Goal 19 WPF application, manual workflow, composition, model audit,
  packaging, localization, and experimental model files listed in
  `docs/GOAL-19-COMPLETION.md`.
- `Directory.Build.props` for version `0.0.20`.
- `src/GraphReader.App/Services/ManualPreviewWorkspaceService.cs`.
- `src/GraphReader.Export/ExportSerialization.cs`.
- `tests/GraphReader.Export.Tests/ExportServiceAcceptanceTests.cs`.
- `tests/GraphReader.Integration.Tests/IntegrationSmoke/ManualPreviewWorkflowSmokeTests.cs`.
- `tests/GraphReader.Integration.Tests/IntegrationSmoke/RealGraphManualPortableValidationTests.cs`.
- `packaging/portable-validation/`.
- `docs/GOAL-19-COMPLETION.md`, `docs/1.0-READINESS.md`, and this record.

Private images, generated evidence, model weights, training outputs, recovery
archives, and build output remain ignored and untracked.

## Tests and commands

- Opt-in Chandler WPF validation: 1/1 passed.
- Full Release solution: 565 passed, 4 deliberately opt-in local-asset tests
  skipped, and 0 failed. An initial run exposed one transient Windows cache
  overwrite race; the exact failing test passed in isolation and the complete
  unchanged suite then passed.
- Full integration assembly without private environment variables: 76 passed,
  1 correctly skipped opt-in test, 0 failed.
- Manual phase-semantics focus: 1/1 passed.
- Portable isolation self-tests: 10/10 in Windows PowerShell and 10/10 in
  PowerShell 7.
- Live portable isolation: 7/7 gates passed.
- Export serialization suite: 18/18 passed, including spreadsheet formula
  prefix protection and unchanged negative numeric values.
- Release-artifact packaging regression: 43/43 passed.
- Development-portable regression: 6/6 passed.
- Localization behavior: 9/9 passed; repository audit found 156 keys, zero
  missing, extra, duplicate, or unresolved references.
- Public synthetic scoreboard: 36/36 gates passed in 234.938 ms with peak
  managed memory of 692,176 bytes. This is metric-contract smoke, not detector
  accuracy.
- `Build-Windows.ps1 -AuditOnly`: release not ready, 12 blockers, 6 manifests,
  0 redistributable model files, and no artifacts emitted.
- `git diff --check`: passed.

## Metrics and timing

- Chandler WPF validation: 1 test passed in the parent rerun.
- Source image: 863 by 395 pixels, 36,222 bytes, unchanged SHA-256.
- Manual result: 2 series, 6 points, 2 dividers, 3 calibration anchors.
- Rendered WPF evidence: 1400 by 900 pixels, 103,745 bytes.
- Portable isolation live harness: approximately 10 seconds, 7/7 gates.

## License review

The Chandler image is private, local-only validation input with no
redistribution authorization. It is not tracked, trained on, or packaged. No
new dependency, model, binary, or license approval was introduced by Goal 20.
All Goal 19 production blockers remain fail closed, including OpenCV Microsoft
runtime attestation and NCNN/Microsoft runtime notice provenance.

## Commit and push

The required checkpoint commit subject is
`Add manual portable preview and Goal 19 audit groundwork`. The checkpoint is
integrated into `main` and pushed to `origin/main` only after all validations
listed here pass. No tag, installer, release ZIP, GitHub release, or public
artifact is created.

## Portable artifact

After the checkpoint commit, the development portable is rebuilt from that
exact clean commit. Its ignored `latest.json` is the authoritative local record
for the exact commit, version, payload, and launch command. This is a
development portable, not either mandatory public release artifact.

## Remaining production blockers

- All six automatic stages remain unavailable in production composition.
- No complete approved, redistributable, checksum-verified production model
  set exists.
- OCR and marker quality gates remain failed.
- OpenCV and NCNN transitive redistribution evidence remains incomplete.
- No clean Windows VM installer/portable parity, repair, upgrade, uninstall,
  offline, or clean-profile evidence exists.
- No public installer or portable ZIP is authorized.

## Acceptance status

- Goal 19 recovery checkpoint: **PASS**.
- Real manual portable workflow: **PASS**.
- Purpose-aware local portable isolation: **PASS** with stated local-evidence
  limitations.
- Automatic production workflow: **FAIL**.
- `1.0.1` readiness: **FAIL**.
