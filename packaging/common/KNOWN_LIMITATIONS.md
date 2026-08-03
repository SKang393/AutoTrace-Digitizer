<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Known limitations

- Windows x64 is the only supported release runtime.
- The first release has no automatic updater.
- Project files created by a newer build are not guaranteed to open in an
  older build. Installer downgrade is blocked by default.
- Optional model or hardware-provider behavior remains unavailable unless its
  manifest, checksum, redistribution terms, notices, and benchmark review pass
  the release audit.
- Portable sentinel/path-policy code exists, but application-wide consumption,
  read-only-folder diagnostics, and clean-profile path isolation are not yet
  runtime verified.
- Installer launch, repair, upgrade, self-uninstall, Start Menu, and user-data
  preservation still require clean-machine runtime evidence.
- Clean-machine visual and workflow checks remain mandatory release evidence.
