<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Common Windows publish stage

`publish.json` defines the single self-contained `win-x64` application publish
used by both Windows distributions. `Build-Windows.ps1` reads the application
version only from `Directory.Build.props`, then copies this publish without
rebuilding the application for either distribution.

The build stages contracts, model manifests, Apache licensing, third-party
notices, and complete third-party license texts into the common publish before
copying that same content into both distribution staging directories.
Checksum-approved model artifacts are copied to their manifest-resolved paths:
an existing `models/...` path is preserved, while a bare relative path is
placed beneath `models/runtime/`. Source and archive resolution must each be
unique, and the staged file is rehashed before distribution staging.

The build also adds `build-metadata.json` to the shared payload and emits
release notes, known limitations, a CycloneDX SBOM, SHA-256 checksums, and
release metadata beside the final artifacts. `-AuditOnly` performs the release
eligibility, provenance, license, and model checks without publishing or
writing release output.

`release-audit.json` is the tracked, machine-readable release authority. It
contains exact component fields and ordered first-match binary coverage rules.
The ignored local dependency ledger is only an optional discrepancy signal and
cannot make a clean checkout fail merely because it is absent.

The required-content list is an exact source-to-target allowlist. The build
also records the application publish file set, rejects reserved-path
collisions, and requires the common, installer, and portable payloads to match
their exact allowed paths before creating artifacts.
Artifact emission always performs a fresh self-contained publish;
`-SkipPublish` is rejected outside audit-only mode.

After artifact creation, the build invokes the deep release verifier before it
can return `ArtifactsEmitted = true`. This rechecks payload identity, manifests,
notices, SBOM coverage, checksums, portable isolation metadata, and the
installer's side-effect-free embedded-payload verification command.
