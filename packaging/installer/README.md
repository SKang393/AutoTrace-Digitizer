<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Windows installer definition

`installer.json` records the enforceable installer behavior. The Apache-2.0
`GraphReader.Installer` bootstrapper is per-user, does not require elevation,
and embeds an archive of the common publish stage without rebuilding the
application binaries.

The installer writes binaries beneath
`%LOCALAPPDATA%\Programs\GraphAutoReader`, creates a Start Menu shortcut and a
normal current-user uninstall entry, and leaves mutable data beneath
`%LOCALAPPDATA%\GraphAutoReader` intact during uninstall. Same-version install
repairs files, newer versions upgrade, and downgrade is blocked unless the
operator explicitly passes `--allow-downgrade`.

Clean-machine install, launch, repair, upgrade, uninstall, and user-data
preservation remain mandatory evidence before publication. The tracked
definition remains `implementation-complete-runtime-unverified` until that
evidence exists.
