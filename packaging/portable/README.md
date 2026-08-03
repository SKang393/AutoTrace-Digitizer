<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Windows portable definition

`portable.json` records the enforceable portable behavior. The build
script creates an empty `portable.mode` beside `GraphReader.App.exe` in
portable staging. The domain path policy resolves that sentinel to `Data`
beneath the extracted application directory. End-to-end application
composition has not yet proved that every mutable subsystem consumes that path
policy or reports a read-only extraction directory clearly. Runtime sentinels
are generated and remain ignored by Git.

The ZIP is created directly from that staging directory and retains root-level
application, license, contract, manifest, and build-metadata files. Publication
still requires read-only-folder handling, registry independence, path
isolation, read-only handling, and clean-machine launch checks from paths
containing spaces and Unicode characters. Application integration and
read-only handling remain Session 18 work, so the tracked definition is
`packaging-implemented-application-integration-pending`.
