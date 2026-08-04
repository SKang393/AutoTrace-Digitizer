<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Goal 19 completion record

## Problem

The existing internal build had no maintainable live portable preview, ordinary
application composition depended on recorded data or an all-or-nothing
production failure, and production release inputs lacked approved models,
complete OpenCvSharp native provenance, private supplied-example evidence, and
clean-machine distribution evidence.

## Implementation

- Added a real-data WPF manual workflow for image import, three-anchor
  calibration, series creation, editable points and phase dividers, explicit
  shared-baseline/probe relations, save, autosave, recovery, reopen, and CSV
  export.
- Kept printed X, estimated X, observation order, and modification history
  separate and auditable.
- Added deterministic marker symbols, unknown-fill rendering, localization,
  zoom/pan controls, empty-state magnifier behavior, and failure-safe commands.
- Replaced all-or-nothing production composition with a manual-capable,
  stage-aware registry. Automatic stages remain disabled unless their complete
  production evidence is approved.
- Added atomic development-portable build, launch, and watcher scripts.
- Added independent packaging and verification for multi-file model payloads.
- Added a bounded WPF composition regression that exercises the real manual
  ViewModel commands, canvas edits, save, recovery, reopen, and export without
  selecting recorded fake graph data.
- Added a fail-closed local portable validation harness for Unicode and space
  paths, shared and normal portable data roots, endpoint observation, registry
  observation, read-only diagnostics, and file-system tracing.

## Portable preview

Goal 19 historical local checkpoint, superseded by the Goal 20 recovery record:

- version: `0.0.19`
- base commit: `616dcf5c792b0f89b9c45311fb03abbed770513e`
- dirty development build: yes
- build time: `2026-08-03T22:59:35.3760356+00:00`
- executable:
  `artifacts/dev-portable/builds/0.0.19-20260803T225935376Z-616dcf5c/GraphReader.App.exe`
- running process: `66708`, responsive, window title `Graph Auto Reader`
- fast tests: PASS, App 34/34, Domain 23/23, focused integration 36/36
- full tests: PASS, 567/567 with exact ignored marker and OCR experiments
- production-model IDs reported available: `0`
- automatic stages reported unavailable: `6`
- mutable root: `artifacts/dev-portable/Data`
- mutable entries after empty launch: `Autosave`, `Cache`, `Logs`, `Recovery`,
  and `Settings` directories; no graph/project/export file
- retained empty-state screenshot from the byte-identical prior payload:
  `artifacts/dev-portable/evidence/manual-preview-empty.png`
- retained screenshot SHA-256:
  `56f56e9d2d4cfebea01acb728f051893130c7ea636d707a0093b29861ab82ef4`

Start the watcher with
`powershell.exe -File packaging/Watch-DevPortable.ps1 -BuildOnStart -FastTestsOnly -AllowDirty -LaunchAfterBuild`.
Launch the latest stable build with
`packaging/Run-Latest-DevPortable.cmd`. Import, tab management, manual
calibration, series and point editing, phase editing, project save, autosave,
recovery, reopen, and CSV export are usable. All six automatic stages remain
unavailable.

The Goal 20 recovery promoted this internal checkpoint to `0.0.20` only after
the user supplied the private Chandler graph. The real `ManualPreview` WPF
composition then passed import, calibration, manual edits, phase editing,
save/reopen, export, autosave/recovery, zoom, pan, fixed magnifier, and rendered
window evidence without fake graph data. Purpose-aware portable isolation also
passed 7/7 local gates while retaining attributed Windows, WPF, .NET, and GPU
cache activity as evidence-bearing warnings. See `docs/GOAL-20-COMPLETION.md`.

This remains local isolated-profile evidence, not clean-profile or clean-VM
evidence.

## Production adapters

Import, manual calibration, review, save, recovery, and export remain available
in Production composition. Automatic stages currently report:

| Stage | State | Exact blocker |
| --- | --- | --- |
| Enhancement | Unavailable | No installed approved Real-ESRGAN runtime, payload, and Graph Auto Reader benchmark set. |
| Axis | Unavailable | Deterministic adapter exists; OpenCvSharp native redistribution audit is blocked. |
| OCR | Unavailable | Detection and general recognition remain metadata-only; both V1 CTC runs and both distinct V2 spatial-sequence runs failed quality gates. |
| Markers | Unavailable | Marker-center acceptance failed and the classifier candidate missed its shape gate. |
| Legends | Unavailable | Deterministic reasoner requires approved OCR and marker evidence. |
| Phases | Unavailable | Deterministic reasoner requires approved axis, OCR, and marker evidence. |

No automatic stage executes in Production with candidate assets.

## Models and provenance

`models/manifest/PRODUCTION_MODEL_MATRIX.md` records exact current candidates.
There are six manifests and zero installed, checksum-verified,
release-eligible model files.

Bounded ignored development evidence improved the model decisions without
approving any model:

- A release-minimal Real-ESRGAN anime x2 runtime excluded the nonredistributable
  debug OpenMP DLL, demos, and unused models. It passed 2/2 real-adapter Vulkan
  runs at exact 2x dimensions in 2366.1442 ms and 1046.5886 ms, with a
  73.0172 ms cache hit. Required graph accuracy, memory, CPU fallback,
  complete notices, and authorized `vcomp140.dll` provenance remain open.
- The official `RealESRGAN_x2plus` hash was reverified, but its PyTorch
  checkpoint is incompatible with the current NCNN adapter and no conversion
  is authorized.
- The marker-center artifact reproduced byte for byte with validation F1@5px
  `1.0000`, but its historical exact held-out gate remains failed at 5/6
  fixtures.
- The new marker-classifier candidate reproduced byte for byte and passed an
  exact CPU/DirectML probe. Validation shape macro-F1 is `0.871062`, below the
  local `0.90` gate, and the packed wrapper lacks a direct sealed held-out run.
- The project-trained graph-numeric CTC experiment used only procedural vector
  glyphs. Candidate and single repair held-out exact match were both
  `0.015625`; the repair CER worsened to `0.932710`. Its exact ONNX passed an
  execution-only CPU/DirectML probe, but no OCR manifest or approval was made.
- A distinct dense spatial-sequence V2 experiment used disjoint renderer,
  glyph-family, and degradation families. Its one-factor contrast repair
  improved held-out exact match from `0.046875` to `0.10546875`, but held-out
  CER remained `1.201399`. The exact repair ONNX passed CPU and DirectML tensor
  execution, but no manifest or approval was created.
- Official PaddleOCR metadata is pinned to source revision
  `33cbdd9deb2e00f61e7966db70669b249c005a37`. The separately hosted archives
  still lack artifact-specific redistribution evidence, immutable artifact
  revisions, and published SHA-256 values. The exact repository dictionary is
  recorded, but it does not approve the uninspected archive.

The exact OpenCvSharp slim package remains blocked:

- native package SHA-256:
  `281551a6c032d1aab316db9c1817bcded5a85188b24b2efd12c02665e7233817`
- `OpenCvSharpExtern.dll` SHA-256:
  `1fa122bdb8e94175e7719fb8aaf2ab211268a756f5d0c7a13c710ed79ae30cd`
- native payload files: 1
- direct imported system DLLs: 9
- included native license/notice/SBOM/link-map files: 0

The binary is statically linked and direct PE imports cannot establish the
complete linked dependency set. No native inventory or third-party notice file
was fabricated. `redistribution: false` and `reviewStatus: blocked` remain.

A separate pinned source-build fallback compiled and passed 3/3 focused
axis-provider tests without altering that decision. Canonical path mapping made
two clean builds byte-identical: the 7,965,696-byte DLL SHA-256 is
`87c12460daba638b36e916ea2bb832d0759fbf094b8639919a7ce11b0cca5791`
and the 16,390,044-byte linker-map SHA-256 is
`e7f9f768b82172b9f2021b2a469de371962655bd0833c8f214bbefdad05a8a77`.
All 15 evidence entries now map to a candidate disposition and four complete
notice sections validate. Five linked Microsoft static-runtime entries still
require explicit release-maintainer license attestation, so the source build
has not replaced the blocked NuGet runtime or changed the public release audit.

## Files changed

- WPF application, controls, localization, composition, runtime paths, models,
  services, and ViewModels under `src/GraphReader.App/`
- application and integration regressions under `tests/`
- development portable scripts and tests under `packaging/`
- Windows build and artifact verifier multi-file model support
- pinned, fail-closed OpenCV source-build scaffold under
  `packaging/opencv-source/`
- candidate-only OpenCV source review policy and notice bundle under
  `packaging/opencv-source/review/`
- fail-closed portable runtime validation under
  `packaging/portable-validation/`
- Real-ESRGAN partial runtime evidence and marker candidate manifests
- failed V1 CTC and V2 spatial-sequence OCR experiment source, tests, and audit
- production model matrix and `docs/1.0-READINESS.md`

Private images, model weights, generated datasets, build output, and evidence
files are not tracked.

## Tests and commands

- `dotnet test GraphAutoReader.slnx -c Release --no-restore` with the exact
  ignored marker-classifier, V1 OCR, and V2 OCR experiments enabled: 567/567
  passed.
- focused manual workflow: 8/8 passed.
- bounded WPF manual composition: 1/1 passed.
- Goal 19 application/composition focus: 18/18 passed in reviewer verification.
- `packaging/tests/Test-DevPortable.Tests.ps1`: 6/6 passed, including
  fail-closed single-file and multi-file model discovery.
- `packaging/tests/Test-ReleaseArtifact.Tests.ps1`: 43/43 passed.
- Super-resolution tests: 57/57 passed.
- Marker Python pipelines: 15/15 passed.
- Marker .NET assembly with the exact ignored candidate: 83/83 passed.
- OCR V1 Python scaffold: 17/17 passed.
- OCR V2 Python tests: 3/3 passed.
- OCR .NET assembly with both exact ignored failed experiments: 82/82 passed.
- OpenCV source-audit behavior tests: 6/6 passed in Windows PowerShell and
  PowerShell 7. Retained lock/cache/triplet inputs, preflight metadata, and both
  CMake toolchain caches are bound fail closed to the tracked source lock.
- Source-built OpenCV focused axis-provider tests: 3/3 passed.
- OpenCV repeat-build comparison: passed once; DLL and linker-map hashes match.
- OpenCV candidate review policy and negative-path test: passed in both
  PowerShell runtimes; overall status remains
  `requires-maintainer-attestation`.
- Portable validation self-tests: 7/7 in Windows PowerShell and PowerShell 7.
- Live local portable validation after Goal 20 classification repair: 7/7;
  55 allowed portable `Data` events, 12 attributed external warnings, zero
  application-owned or unattributed failures, and zero watcher errors.
- Public scoreboard: 36/36 synthetic metric-contract gates passed in 196.049 ms.
- `packaging/localization/Test-LocalizationAudit.ps1`: 9/9 passed.
- repository localization audit: 156 keys, 0 missing, 0 extra, 0 duplicate,
  and 0 unresolved references.
- `git diff --check`: passed.
- `Build-Windows.ps1 -AuditOnly`: release not ready, 12 blockers, six
  manifests, zero redistributable model files, and no emitted artifacts.

## Metrics and timing

- full .NET suite: 567 tests passed with no skips
- latest development portable build passed App 34/34, Domain 23/23, focused
  integration 36/36, publish, and process smoke
- watcher debounce: 2.53 to 2.63 seconds across independent verification
- queued watcher behavior: four expected builds
- manual WPF composition: 1 test in 1 second
- OpenCV clean source builds: 161.546 and 171.175 seconds
- OCR V2 candidate and repair training: 221.038 and 199.791 seconds

## License review

No dependency or model approval was inferred from repository-level licensing.
The OpenCvSharp managed package remains reviewed separately from its blocked
native runtime. No GPL, AGPL, SSPL, BUSL, or non-commercial dependency was
added. Only the exact user-authorized official Real-ESRGAN assets and pinned
OpenCV source revisions were imported into ignored local audit storage after
recording source, purpose, checksum, license, privacy, and Git eligibility. No
private graph was imported during Goal 19. Goal 20 later imported the explicitly
user-supplied Chandler image into ignored local evaluation storage only, with
no training, Git, packaging, or redistribution eligibility.

The OpenCV candidate notice bundle covers Apache-2.0, zlib, SoftFloat, and
FDLIBM obligations for the retained minimal source profile. It is not a public
notice approval. Five Microsoft static-runtime entries need maintainer
attestation. The minimal NCNN runtime excludes `vcomp140d.dll`; the remaining
release `vcomp140.dll` still needs an authorized Microsoft source and license
record.

## Readiness changes

- Manual-first development preview: implemented and running.
- Stage-aware production composition: implemented, automatic stages fail closed.
- Multi-file param/bin packaging limitation: resolved and verified.
- OpenCV source reproducibility and candidate notice classification: resolved.
- OpenCV release approval: still blocked on Microsoft runtime attestation.
- Local portable path, diagnostic, endpoint, registry-key, and write tracing:
  recorded with one fail-closed isolation gate.
- Current release audit: 12 legitimate blockers. The false single-file-only
  model restriction is removed; the new classifier manifest adds a truthful
  missing-candidate-payload blocker.
- Version promotion: intentionally unchanged at `0.0.19`.
- Public release decision: unchanged, `FAIL`.

## Known limitations

- The user-authorized private Chandler graph now has complete manual-only Goal
  20 evidence. It does not provide automatic detector accuracy evidence.
- No production model set is approved or installed. Bounded Real-ESRGAN and
  marker evidence remains experimental and ignored. All four bounded numeric
  OCR runs failed; the V1 and V2 experiment budgets are exhausted.
- OpenCvSharp source inventory and candidate notices are classified, but five
  Microsoft static-runtime attestations and public release integration remain
  open.
- The minimal source-built OpenCvSharp fallback has focused functional parity
  and repeat-build parity, but full application benchmarks and clean-machine
  checks remain open.
- A purpose-aware local isolated-profile simulation passed, but it is not a
  clean profile or VM. Network remained enabled and polling can miss brief
  connections. FileSystemWatcher cannot directly identify the writer process.
- No installer/portable release parity, repair, upgrade, or uninstall evidence
  exists.

## Artifact paths

- `artifacts/dev-portable/latest.json`
- `artifacts/dev-portable/builds/0.0.19-20260803T225935376Z-616dcf5c/`
- `artifacts/dev-portable/Data/`
- `artifacts/dev-portable/evidence/manual-preview-empty.png`
- `artifacts/evidence/goal19-track-a/manual-preview-track-a.trx`
- `artifacts/evidence/goal19-wpf-root/manual-preview-wpf-composition-root.trx`
- `.omo/evidence/goal-19-track-a-code-review.md`
- `artifacts/goal19/localization-report.json`
- `artifacts/goal19-realesrgan/adapter-benchmark-minimal/evidence.json`
- `models/manifest/markers/MARKER_CLASSIFIER_CANDIDATE_AUDIT.md`
- `models/manifest/ocr/GRAPH_NUMERIC_CTC_EXPERIMENT_AUDIT.md`
- `models/manifest/ocr/GRAPH_NUMERIC_SEQUENCE_V2_EXPERIMENT_AUDIT.md`
- `artifacts/goal19-opencv-source/evidence-repro-pass2-final-a/`
- `artifacts/goal19-opencv-source/evidence-repro-pass2-final-b/`
- `artifacts/portable-validation/20260803T224522086Z-940006c1/portable-clean-profile-report.json`

## Integration notes

Goal 20 recovered this work, satisfied the mandatory manual real-graph gate,
and prepared the required `0.0.20` checkpoint for integration into `main`. The
original workspace's unrelated dirty files remain untouched. No tag,
installer, portable release ZIP, GitHub release, or `1.0.1` promotion was
created.

## Acceptance status

- Portable preview: **PASS** for the internal manual development checkpoint.
  The user-authorized Chandler workflow and local purpose-aware isolation now
  have direct Goal 20 evidence.
- Production workflow: **FAIL**. Manual composition works, but all six automatic
  stages are unavailable.
- 1.0 readiness: **FAIL**. Mandatory model, provenance, private validation,
  packaging, and clean-machine gates remain open.
