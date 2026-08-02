<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Common Windows publish stage

`publish.json` defines the single self-contained `win-x64` application publish
used by both Windows distributions. Generated files belong under the build
output root selected by `Build-Windows.ps1`; this source directory contains
definitions only.

The build stages contracts, model manifests, Apache licensing, third-party
notices, and complete third-party license texts into the common publish before
copying that same content into both distribution staging directories.
