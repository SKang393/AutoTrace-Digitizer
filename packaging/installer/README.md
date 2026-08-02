<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Windows installer definition

`installer.json` records the enforceable Goal 00 installer behavior. The
installer is per-user, does not require elevation by default, and consumes the
common publish stage without rebuilding application binaries.

Goal 00 deliberately does not select or bundle an installer authoring tool and
does not emit a setup executable. A later release session must select a
license-compatible tool, implement install, repair, and uninstall behavior,
and pass the VM checks in `WINDOWS_DISTRIBUTION.md`.
